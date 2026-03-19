#!/usr/bin/env python3
"""Phase 4: Test figurative descriptions — verify the MARC property."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from db import (
    init_db, get_all_tasks, get_task, get_figurative,
    insert_figurative_trial, update_figurative_response,
    update_figurative_evaluation,
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


def _load_arc(config, arc_name):
    arc_path = resolve_arc_path(config, arc_name)
    if arc_path is None:
        raise FileNotFoundError(f"ARC task file not found for {arc_name}")
    with open(arc_path) as f:
        return json.load(f)


def _serialize_prompt(messages):
    """Serialize messages for storage, stripping binary data."""
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


def _get_extraction_model_config(config, model_name):
    """Get gpt-oss-120b config for extraction if subject model is different."""
    if model_name == "gpt-oss-120b":
        return None  # same model, no override needed
    for m in config["models"]:
        if m["name"] == "gpt-oss-120b":
            return m
    return None


def _run_single_figurative_trial(model_config, config, db_path, fig_id, tid, k,
                                  model_name, is_vision):
    """Run a single figurative trial. Thread-safe: opens its own DB connection."""
    conn = init_db(db_path)
    try:
        task = get_task(conn, tid)
        fig = conn.execute(
            "SELECT * FROM figurative_descriptions WHERE fig_id=?", (fig_id,)
        ).fetchone()

        arc_task = _load_arc(config, task["arc_name"])

        if k == 0:
            condition = "figurative_only"
            messages = build_prompt(condition, task, arc_task, fig_row=fig, vision=is_vision)
        else:
            condition = "figurative_with_examples"
            messages = build_prompt(condition, task, arc_task, num_examples=k,
                                    fig_row=fig, vision=is_vision)

        prompt_text = _serialize_prompt(messages)
        trial_id = insert_figurative_trial(conn, fig_id, tid, model_name, k, 1, prompt_text)

        # Skip if already done
        row = conn.execute(
            "SELECT response_text, error FROM figurative_trials WHERE trial_id=?",
            (trial_id,)
        ).fetchone()
        if row["response_text"] or row["error"]:
            return ("skip", tid, k, "")

        extraction_config = _get_extraction_model_config(config, model_name)
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

            if predicted_grid is not None or attempt == max_attempts - 1:
                break
            # Retry on parse error

        update_figurative_response(conn, trial_id, combined_raw, extracted, total_ms)

        if predicted_grid is not None:
            expected = arc_task["test"][0]["output"]
            correct, cell_acc = compare_grids(predicted_grid, expected)
            update_figurative_evaluation(
                conn, trial_id, json.dumps(predicted_grid), reasoning,
                1 if correct else 0, cell_acc,
            )
            status = "CORRECT" if correct else f"wrong (cell_acc={cell_acc:.2f})"
        else:
            update_figurative_evaluation(conn, trial_id, None, reasoning, 0, 0.0)
            status = f"parse_error: {parse_error}"

        retries = f" (retries={attempt})" if attempt > 0 else ""
        return ("ok", tid, k, f"{status} ({total_ms}ms){retries}")

    except Exception as e:
        try:
            update_figurative_response(conn, trial_id, None, None, 0, error=str(e))
        except Exception:
            pass
        return ("error", tid, k, str(e))
    finally:
        conn.close()


@click.group()
def cli():
    """Phase 4: Test figurative descriptions for the MARC property."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", required=True, help="Model to test with")
@click.option("--generator-model", "generator_model", default="claude-sonnet")
@click.option("--source", default=None, type=click.Choice(["larc", "marc-submission"]),
              help="Filter tasks by source (default: all)")
@click.option("--concurrency", default=1, type=int,
              help="Number of parallel API calls (default: 1, use 8 for MindRouter)")
@click.option("--limit", default=None, type=int)
@click.option("--dry-run", is_flag=True)
def test(config_path, db_path, model_name, generator_model, source, concurrency,
         limit, dry_run):
    """Test figurative descriptions across example counts (k=0,1,...,num_train)."""
    config = load_config(config_path)
    model_config = _find_model_config(config, model_name)
    is_vision = model_config.get("modality") == "vision"

    conn = init_db(db_path)
    tasks = get_all_tasks(conn, source=source)

    # Collect tasks with figurative descriptions
    trial_plan = []
    for task in tasks:
        tid = task["task_id"]
        fig = get_figurative(conn, tid, generator_model)
        if not fig:
            continue
        for k in range(0, task["num_train"] + 1):
            trial_plan.append((tid, fig["fig_id"], k))

    if limit:
        task_ids = []
        limited_plan = []
        for tid, fig_id, k in trial_plan:
            if tid not in task_ids:
                task_ids.append(tid)
            if len(task_ids) <= limit:
                limited_plan.append((tid, fig_id, k))
        trial_plan = limited_plan

    click.echo(f"Testing {len(trial_plan)} trials across figurative descriptions")
    click.echo(f"  Model: {model_name} | Generator: {generator_model} | Concurrency: {concurrency}")

    if dry_run:
        for tid, fig_id, k in trial_plan[:20]:
            click.echo(f"  task={tid} fig={fig_id} k={k}")
        if len(trial_plan) > 20:
            click.echo(f"  ... and {len(trial_plan) - 20} more")
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
                    _run_single_figurative_trial, model_config, config, db_path,
                    fig_id, tid, k, model_name, is_vision
                ): (tid, fig_id, k)
                for tid, fig_id, k in trial_plan
            }
            for i, future in enumerate(as_completed(futures), 1):
                result_type, tid, k, msg = future.result()
                if result_type == "skip":
                    skipped += 1
                elif result_type == "ok":
                    completed += 1
                    click.echo(f"[{i}/{len(trial_plan)}] task={tid} k={k} ... {msg}")
                else:
                    errors += 1
                    click.echo(f"[{i}/{len(trial_plan)}] task={tid} k={k} ... ERROR: {msg}")
                if i % 50 == 0:
                    click.echo(f"  --- progress: {completed} done, {skipped} skipped, {errors} errors ---")
    else:
        # Sequential fallback (worker opens its own conn)
        for i, (tid, fig_id, k) in enumerate(trial_plan, 1):
            result_type, rtid, rk, msg = _run_single_figurative_trial(
                model_config, config, db_path, fig_id, tid, k, model_name, is_vision
            )
            if result_type == "skip":
                skipped += 1
            elif result_type == "ok":
                completed += 1
                click.echo(f"[{i}/{len(trial_plan)}] task={tid} k={k} ... {msg}")
            else:
                errors += 1
                click.echo(f"[{i}/{len(trial_plan)}] task={tid} k={k} ... ERROR: {msg}")

    click.echo(f"\nDone. completed={completed} skipped={skipped} errors={errors}")


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", required=True)
@click.option("--generator-model", "generator_model", default="claude-sonnet")
@click.option("--source", default=None, type=click.Choice(["larc", "marc-submission"]),
              help="Filter tasks by source (default: all)")
def verify_marc(db_path, model_name, generator_model, source):
    """Check which tasks satisfy the MARC property.

    MARC property:
    1. Examples alone fail (baseline examples_only)
    2. Figurative alone fails (k=0)
    3. Figurative + sufficient examples succeeds (some k>0)
    """
    conn = init_db(db_path)
    tasks = get_all_tasks(conn, source=source)

    marc_puzzles = []
    non_marc = []

    for task in tasks:
        tid = task["task_id"]
        fig = get_figurative(conn, tid, generator_model)
        if not fig:
            continue

        # Check condition 1: examples alone must fail
        ex_only = conn.execute(
            """SELECT MAX(correct) as best FROM baseline_trials
               WHERE task_id=? AND model_name=? AND condition='examples_only'""",
            (tid, model_name),
        ).fetchone()
        if ex_only and ex_only["best"] == 1:
            non_marc.append((tid, "examples_alone_succeeds"))
            continue

        # Check condition 2: figurative alone must fail
        fig_only = conn.execute(
            """SELECT correct FROM figurative_trials
               WHERE fig_id=? AND model_name=? AND num_examples=0""",
            (fig["fig_id"], model_name),
        ).fetchone()
        if fig_only and fig_only["correct"] == 1:
            non_marc.append((tid, "figurative_alone_succeeds"))
            continue

        # Check condition 3: figurative + examples must succeed for some k
        fig_success = conn.execute(
            """SELECT MIN(num_examples) as min_k FROM figurative_trials
               WHERE fig_id=? AND model_name=? AND num_examples>0 AND correct=1""",
            (fig["fig_id"], model_name),
        ).fetchone()
        if fig_success and fig_success["min_k"] is not None:
            marc_puzzles.append((tid, fig_success["min_k"], fig["metaphor"]))
        else:
            non_marc.append((tid, "figurative_never_succeeds"))

    click.echo(f"\n=== MARC Property Verification ({model_name}) ===\n")
    click.echo(f"Valid MARC puzzles: {len(marc_puzzles)}")
    for tid, min_k, metaphor in marc_puzzles:
        click.echo(f"  task={tid} n={min_k}: \"{metaphor[:60]}\"")

    click.echo(f"\nNon-MARC: {len(non_marc)}")
    reasons = {}
    for tid, reason in non_marc:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        click.echo(f"  {reason}: {count}")

    conn.close()


if __name__ == "__main__":
    cli()
