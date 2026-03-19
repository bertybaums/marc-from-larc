#!/usr/bin/env python3
"""Phase 1: Baseline 3-condition testing across models and tasks.

Uses a two-pass approach:
  Pass 1: Let the model reason freely about the puzzle
  Pass 2: Extract the structured JSON answer from the reasoning
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from db import (
    init_db, get_all_tasks, get_task, insert_baseline_trial,
    update_baseline_response, update_baseline_evaluation,
)
from grids import parse_response_grid, compare_grids
from models import call_llm_two_pass
from prompts import build_prompt, build_extraction_messages
from tasks import load_config, resolve_arc_path


def _find_model_config(config, model_name):
    for m in config["models"]:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not found in config")


def _get_extraction_model_config(config, model_name):
    """Get gpt-oss-120b config for extraction if subject model is different."""
    if model_name == "gpt-oss-120b":
        return None
    for m in config["models"]:
        if m["name"] == "gpt-oss-120b":
            return m
    return None


def _load_arc(config, arc_name):
    arc_path = resolve_arc_path(config, arc_name)
    if arc_path is None:
        raise FileNotFoundError(f"ARC task file not found for {arc_name}")
    with open(arc_path) as f:
        return json.load(f)


def _serialize_prompt(messages):
    """Serialize messages to a string for storage (strip binary image data)."""
    cleaned = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            parts = []
            for block in msg["content"]:
                if block.get("type") == "image_url":
                    parts.append({"type": "image_url", "image_url": {"url": "[base64 image]"}})
                else:
                    parts.append(block)
            cleaned.append({**msg, "content": parts})
        else:
            cleaned.append(msg)
    return json.dumps(cleaned, ensure_ascii=False)


def _run_single_baseline_trial(model_config, extraction_config, config, db_path,
                                task_id, cond, num_ex, rep, model_name, is_vision):
    """Run a single baseline trial. Thread-safe: opens its own DB connection."""
    conn = init_db(db_path)
    try:
        task = get_task(conn, task_id)
        arc_task = _load_arc(config, task["arc_name"])

        messages = build_prompt(cond, task, arc_task, num_examples=num_ex, vision=is_vision)
        prompt_text = _serialize_prompt(messages)

        trial_id = insert_baseline_trial(
            conn, task_id, model_name, cond, num_ex, rep, prompt_text
        )

        # Skip if already done
        row = conn.execute(
            "SELECT response_text, error FROM baseline_trials WHERE trial_id=?",
            (trial_id,)
        ).fetchone()
        if row["response_text"] or row["error"]:
            return ("skip", task_id, cond, rep, "")

        max_attempts = 3
        for attempt in range(max_attempts):
            raw1, reasoning, raw2, extracted, total_ms = call_llm_two_pass(
                model_config, messages, build_extraction_messages,
                extraction_model_config=extraction_config
            )
            combined_raw = json.dumps({"pass1": json.loads(raw1) if raw1 else None,
                                        "pass2": json.loads(raw2) if raw2 else None})

            predicted_grid, _, parse_error = parse_response_grid(extracted)
            if predicted_grid is None:
                predicted_grid, _, _ = parse_response_grid(reasoning)
                if predicted_grid is not None:
                    parse_error = None

            if predicted_grid is not None or attempt == max_attempts - 1:
                break

        update_baseline_response(conn, trial_id, combined_raw, extracted, total_ms)

        if predicted_grid is not None:
            expected = arc_task["test"][0]["output"]
            correct, cell_acc = compare_grids(predicted_grid, expected)
            update_baseline_evaluation(
                conn, trial_id, json.dumps(predicted_grid), reasoning,
                1 if correct else 0, cell_acc,
            )
            status = "CORRECT" if correct else f"wrong (cell_acc={cell_acc:.2f})"
        else:
            update_baseline_evaluation(conn, trial_id, None, reasoning, 0, 0.0)
            status = f"parse_error: {parse_error}"

        retries = f" (retries={attempt})" if attempt > 0 else ""
        return ("ok", task_id, cond, rep, f"{status} ({total_ms}ms){retries}")

    except Exception as e:
        try:
            update_baseline_response(conn, trial_id, None, None, 0, error=str(e))
        except Exception:
            pass
        return ("error", task_id, cond, rep, str(e))
    finally:
        conn.close()


@click.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_filter", required=True, help="Model name from config")
@click.option("--condition", "cond_filter", default=None,
              type=click.Choice(["examples_only", "language_only", "both"]),
              help="Run only this condition (default: all three)")
@click.option("--limit", default=None, type=int, help="Limit to first N tasks")
@click.option("--repeat", "num_repeats", default=1, type=int, help="Number of repeats per trial")
@click.option("--source", default=None, type=click.Choice(["larc", "marc-submission"]),
              help="Filter tasks by source (default: all)")
@click.option("--concurrency", default=1, type=int,
              help="Number of parallel API calls (default: 1, use 8 for MindRouter)")
@click.option("--dry-run", is_flag=True, help="Print trial plan without calling APIs")
def collect(config_path, db_path, model_filter, cond_filter, limit, num_repeats,
            source, concurrency, dry_run):
    """Run baseline collection: test each task under specified conditions.

    Uses a two-pass approach: first the model reasons freely, then we extract
    the structured answer from its reasoning.
    """
    config = load_config(config_path)
    model_config = _find_model_config(config, model_filter)
    extraction_config = _get_extraction_model_config(config, model_filter)
    is_vision = model_config.get("modality") == "vision"

    conditions = [cond_filter] if cond_filter else ["examples_only", "language_only", "both"]

    conn = init_db(db_path)
    tasks = get_all_tasks(conn, source=source)
    if limit:
        tasks = tasks[:limit]

    # Build trial plan
    trials = []
    for task in tasks:
        for cond in conditions:
            num_ex = task["num_train"] if cond != "language_only" else 0
            for rep in range(1, num_repeats + 1):
                trials.append((task["task_id"], cond, num_ex, rep))

    click.echo(f"Model: {model_filter} | Conditions: {conditions} | "
               f"Tasks: {len(tasks)} | Trials: {len(trials)} | Concurrency: {concurrency}")

    if dry_run:
        for task_id, cond, num_ex, rep in trials[:20]:
            click.echo(f"  task={task_id} cond={cond} examples={num_ex} repeat={rep}")
        if len(trials) > 20:
            click.echo(f"  ... and {len(trials) - 20} more")
        conn.close()
        return

    conn.close()  # Close main conn; workers open their own

    completed = 0
    skipped = 0
    errors = 0

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    _run_single_baseline_trial, model_config, extraction_config,
                    config, db_path, task_id, cond, num_ex, rep, model_filter, is_vision
                ): (task_id, cond, rep)
                for task_id, cond, num_ex, rep in trials
            }
            for i, future in enumerate(as_completed(futures), 1):
                result_type, task_id, cond, rep, msg = future.result()
                if result_type == "skip":
                    skipped += 1
                elif result_type == "ok":
                    completed += 1
                    click.echo(f"[{i}/{len(trials)}] task={task_id} {cond} r={rep} ... {msg}")
                else:
                    errors += 1
                    click.echo(f"[{i}/{len(trials)}] task={task_id} {cond} r={rep} ... ERROR: {msg}")
                if i % 50 == 0:
                    click.echo(f"  --- progress: {completed} done, {skipped} skipped, {errors} errors ---")
    else:
        for task_id, cond, num_ex, rep in trials:
            result_type, rtid, rcond, rrep, msg = _run_single_baseline_trial(
                model_config, extraction_config, config, db_path,
                task_id, cond, num_ex, rep, model_filter, is_vision
            )
            if result_type == "skip":
                skipped += 1
            elif result_type == "ok":
                completed += 1
                click.echo(f"task={task_id} {cond} r={rep} ... {msg}")
            else:
                errors += 1
                click.echo(f"task={task_id} {cond} r={rep} ... ERROR: {msg}")

    click.echo(f"\nDone. completed={completed} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    collect()
