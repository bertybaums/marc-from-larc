#!/usr/bin/env python3
"""Phase 2: Identify task subsets and find minimum examples needed."""

import json

import click

from db import (
    init_db, get_all_tasks, get_task, upsert_subset, get_subsets,
    insert_baseline_trial, update_baseline_response, update_baseline_evaluation,
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


def _get_baseline_result(conn, task_id, model_name, condition):
    """Get whether the baseline condition was correct (any repeat)."""
    row = conn.execute(
        """SELECT MAX(correct) as best_correct FROM baseline_trials
           WHERE task_id=? AND model_name=? AND condition=?""",
        (task_id, model_name, condition),
    ).fetchone()
    if row is None or row["best_correct"] is None:
        return None
    return row["best_correct"] == 1


@click.group()
def cli():
    """Phase 2: Subset identification and minimum-examples search."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", required=True)
def classify(config_path, db_path, model_name):
    """Classify tasks into subsets based on baseline results."""
    conn = init_db(db_path)
    tasks = get_all_tasks(conn)

    counts = {"examples_sufficient": 0, "language_sufficient": 0,
              "both_required": 0, "unsolvable": 0, "incomplete": 0}

    for task in tasks:
        tid = task["task_id"]
        ex_only = _get_baseline_result(conn, tid, model_name, "examples_only")
        lang_only = _get_baseline_result(conn, tid, model_name, "language_only")
        both = _get_baseline_result(conn, tid, model_name, "both")

        if ex_only is None or lang_only is None or both is None:
            counts["incomplete"] += 1
            continue

        if ex_only:
            subset = "examples_sufficient"
        elif lang_only:
            subset = "language_sufficient"
        elif both:
            subset = "both_required"
        else:
            subset = "unsolvable"

        upsert_subset(conn, tid, model_name, subset)
        counts[subset] += 1

    click.echo(f"Subset classification for {model_name}:")
    for name, count in counts.items():
        click.echo(f"  {name}: {count}")

    conn.close()


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", required=True)
@click.option("--dry-run", is_flag=True)
def find_min_examples(config_path, db_path, model_name, dry_run):
    """For language_sufficient and both_required tasks, find minimum training examples needed.

    Tests k=1,2,...,num_train with literal description + k examples.
    m = smallest k where model succeeds.
    """
    config = load_config(config_path)
    model_config = _find_model_config(config, model_name)
    is_vision = model_config.get("modality") == "vision"

    conn = init_db(db_path)
    subsets = get_subsets(conn, model_name=model_name)
    candidates = [s for s in subsets if s["subset"] in ("language_sufficient", "both_required")]

    click.echo(f"Finding min examples for {len(candidates)} tasks ({model_name})")

    if dry_run:
        for s in candidates:
            task = get_task(conn, s["task_id"])
            click.echo(f"  task={s['task_id']} subset={s['subset']} num_train={task['num_train']}")
        return

    for s in candidates:
        task = get_task(conn, s["task_id"])
        tid = task["task_id"]

        try:
            arc_task = _load_arc(config, task["arc_name"])
        except FileNotFoundError as e:
            click.echo(f"  SKIP task={tid}: {e}")
            continue

        min_k = None
        for k in range(1, task["num_train"] + 1):
            # Check if we already have this trial
            existing = conn.execute(
                """SELECT correct FROM baseline_trials
                   WHERE task_id=? AND model_name=? AND condition='both'
                   AND num_examples=? AND repeat_num=1""",
                (tid, model_name, k),
            ).fetchone()

            if existing and existing["correct"] is not None:
                if existing["correct"] == 1:
                    min_k = k
                    break
                continue

            # Run trial with k examples
            messages = build_prompt("both", task, arc_task, num_examples=k, vision=is_vision)
            prompt_text = json.dumps(messages, ensure_ascii=False, default=str)

            trial_id = insert_baseline_trial(conn, tid, model_name, "both", k, 1, prompt_text)

            click.echo(f"  task={tid} k={k}/{task['num_train']} ... ", nl=False)

            try:
                raw1, reasoning, raw2, extracted, total_ms = call_llm_two_pass(
                    model_config, messages, build_extraction_messages
                )
                combined_raw = json.dumps({"pass1": json.loads(raw1) if raw1 else None,
                                            "pass2": json.loads(raw2) if raw2 else None})
                update_baseline_response(conn, trial_id, combined_raw, extracted, total_ms)

                predicted_grid, _, parse_error = parse_response_grid(extracted)
                if predicted_grid is None:
                    predicted_grid, _, _ = parse_response_grid(reasoning)
                if predicted_grid is not None:
                    expected = arc_task["test"][0]["output"]
                    correct, cell_acc = compare_grids(predicted_grid, expected)
                    update_baseline_evaluation(
                        conn, trial_id, json.dumps(predicted_grid), reasoning,
                        1 if correct else 0, cell_acc,
                    )
                    if correct:
                        min_k = k
                        click.echo(f"CORRECT → m={k}")
                        break
                    else:
                        click.echo(f"wrong (cell_acc={cell_acc:.2f})")
                else:
                    update_baseline_evaluation(conn, trial_id, None, reasoning, 0, 0.0)
                    click.echo(f"parse_error")
            except Exception as e:
                update_baseline_response(conn, trial_id, None, None, 0, error=str(e))
                click.echo(f"ERROR: {e}")

        if min_k is not None:
            upsert_subset(conn, tid, model_name, s["subset"], min_examples=min_k)
            click.echo(f"  task={tid}: m={min_k}")
        else:
            click.echo(f"  task={tid}: no k worked (keeping subset={s['subset']})")

    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", default=None)
def report(db_path, model_name):
    """Print subset summary."""
    conn = init_db(db_path)
    subsets = get_subsets(conn, model_name=model_name)

    by_model = {}
    for s in subsets:
        mn = s["model_name"]
        if mn not in by_model:
            by_model[mn] = {}
        sub = s["subset"]
        if sub not in by_model[mn]:
            by_model[mn][sub] = []
        by_model[mn][sub].append(s)

    for mn, subs in sorted(by_model.items()):
        click.echo(f"\n{mn}:")
        for sub, rows in sorted(subs.items()):
            m_values = [r["min_examples_with_language"] for r in rows
                        if r["min_examples_with_language"] is not None]
            m_info = f" (m: {min(m_values)}-{max(m_values)})" if m_values else ""
            click.echo(f"  {sub}: {len(rows)}{m_info}")

    conn.close()


@cli.command("classify-marc-submissions")
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", required=True)
def classify_marc_submissions(config_path, db_path, model_name):
    """Classify MARC submission tasks (no literal-description conditions).

    For submissions, we only have examples_only baseline. Classification:
    - examples_sufficient: examples_only succeeds
    - unsolvable: examples_only fails (candidate for MARC property testing)
    """
    conn = init_db(db_path)
    tasks = conn.execute(
        "SELECT * FROM tasks WHERE source='marc-submission' ORDER BY task_id"
    ).fetchall()

    counts = {"examples_sufficient": 0, "unsolvable": 0, "incomplete": 0}

    for task in tasks:
        tid = task["task_id"]
        ex_only = _get_baseline_result(conn, tid, model_name, "examples_only")

        if ex_only is None:
            counts["incomplete"] += 1
            continue

        if ex_only:
            subset = "examples_sufficient"
        else:
            subset = "unsolvable"

        upsert_subset(conn, tid, model_name, subset)
        counts[subset] += 1

    click.echo(f"MARC submission classification for {model_name}:")
    for name, count in counts.items():
        click.echo(f"  {name}: {count}")

    conn.close()


if __name__ == "__main__":
    cli()
