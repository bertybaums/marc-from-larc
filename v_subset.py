#!/usr/bin/env python3
"""Vision subset classification.

Simplified: since we only test examples_only, classification is binary:
- examples_sufficient: vision model solves with examples alone
- examples_insufficient: candidate for MARC testing
"""

import click

from db import init_db, get_all_tasks, upsert_subset


def _get_baseline_result(conn, task_id, model_name):
    """Get whether examples_only baseline was correct (any repeat)."""
    row = conn.execute(
        """SELECT MAX(correct) as best_correct FROM baseline_trials
           WHERE task_id=? AND model_name=? AND condition='examples_only'""",
        (task_id, model_name),
    ).fetchone()
    if row is None or row["best_correct"] is None:
        return None
    return row["best_correct"] == 1


@click.command()
@click.option("--db", "db_path", default="marc_vision.db")
@click.option("--model", "model_name", required=True)
@click.option("--source", default=None, type=click.Choice(["larc", "marc-submission"]),
              help="Filter tasks by source")
def classify(db_path, model_name, source):
    """Classify tasks based on vision examples_only results.

    examples_sufficient: vision model solves with examples alone (no MARC property)
    examples_insufficient: candidate for MARC property testing
    """
    conn = init_db(db_path)
    tasks = get_all_tasks(conn, source=source)

    counts = {"examples_sufficient": 0, "examples_insufficient": 0, "incomplete": 0}

    for task in tasks:
        tid = task["task_id"]
        result = _get_baseline_result(conn, tid, model_name)

        if result is None:
            counts["incomplete"] += 1
            continue

        if result:
            subset = "examples_sufficient"
        else:
            subset = "examples_insufficient"

        upsert_subset(conn, tid, model_name, subset)
        counts[subset] += 1

    click.echo(f"Vision subset classification for {model_name}:")
    for name, count in counts.items():
        click.echo(f"  {name}: {count}")

    conn.close()


if __name__ == "__main__":
    classify()
