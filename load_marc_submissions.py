#!/usr/bin/env python3
"""Load MARC submissions into the pipeline.

Scans marc-submissions/*.json, normalizes grid cell values to integers,
writes ARC-format JSON to data/marc-submissions/, and inserts tasks +
figurative descriptions into the database.

Task IDs: 1000 + index (M1=1000, M2=1001, ..., M177=1176).
"""

import json
from pathlib import Path

import click

from db import init_db, upsert_task, insert_figurative


SUBMISSIONS_DIR = Path(__file__).parent / "marc-submissions"
OUTPUT_DIR = Path(__file__).parent / "data" / "marc-submissions"
TASK_ID_OFFSET = 1000


def _normalize_grid(grid):
    """Convert all cell values to integers (string '0' → int 0)."""
    return [[int(cell) for cell in row] for row in grid]


def _normalize_submission(data):
    """Normalize a MARC submission to ARC format with integer cells."""
    normalized = {
        "train": [
            {"input": _normalize_grid(ex["input"]),
             "output": _normalize_grid(ex["output"])}
            for ex in data["train"]
        ],
        "test": [
            {"input": _normalize_grid(ex["input"]),
             "output": _normalize_grid(ex["output"])}
            for ex in data["test"]
        ],
    }
    return normalized


@click.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--dry-run", is_flag=True)
def load(db_path, dry_run):
    """Load MARC submissions into the database."""
    if not SUBMISSIONS_DIR.exists():
        click.echo(f"ERROR: Submissions directory not found: {SUBMISSIONS_DIR}")
        return

    # Collect and sort submission files for deterministic ordering
    files = sorted(SUBMISSIONS_DIR.glob("*.json"))
    click.echo(f"Found {len(files)} submission files")

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        conn = init_db(db_path)

    loaded_tasks = 0
    loaded_figs = 0
    errors = []

    for idx, filepath in enumerate(files):
        task_id = TASK_ID_OFFSET + idx
        arc_name = filepath.stem  # filename without .json

        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            errors.append((filepath.name, str(e)))
            click.echo(f"  ERROR {filepath.name}: {e}")
            continue

        # Extract metaphor (fall back to title)
        metaphor = data.get("metaphor", data.get("title", ""))
        if not metaphor:
            errors.append((filepath.name, "no metaphor or title field"))
            click.echo(f"  SKIP {filepath.name}: no metaphor or title")
            continue

        # Normalize to ARC format
        try:
            normalized = _normalize_submission(data)
        except (ValueError, TypeError, KeyError) as e:
            errors.append((filepath.name, f"normalization error: {e}"))
            click.echo(f"  ERROR {filepath.name}: {e}")
            continue

        num_train = len(normalized["train"])

        if dry_run:
            click.echo(f"  task_id={task_id} arc_name={arc_name} "
                        f"train={num_train} metaphor={metaphor[:60]}")
            loaded_tasks += 1
            continue

        # Write normalized ARC-format JSON
        out_path = OUTPUT_DIR / f"{arc_name}.json"
        with open(out_path, "w") as f:
            json.dump(normalized, f, separators=(",", ":"))

        # Insert task
        upsert_task(
            conn, task_id, arc_name, "metaphor",
            see=metaphor,  # store metaphor in see for display
            do="", grid="",
            num_train=num_train,
            source="marc-submission",
        )
        loaded_tasks += 1

        # Insert figurative description with generator_model='human'
        fig_id = insert_figurative(
            conn, task_id, "human",
            metaphor=metaphor,
            fig_see="",
            fig_do="",
            fig_grid="",
            generation_prompt=None,
            raw_generation=json.dumps({"source_file": filepath.name}),
        )
        if fig_id:
            loaded_figs += 1

        click.echo(f"  [{idx+1}/{len(files)}] task_id={task_id} {arc_name}: "
                    f"\"{metaphor[:50]}\" (train={num_train})")

    click.echo(f"\nLoaded {loaded_tasks} tasks, {loaded_figs} figurative descriptions")
    if errors:
        click.echo(f"Errors: {len(errors)}")
        for name, err in errors:
            click.echo(f"  {name}: {err}")

    if not dry_run:
        conn.close()


if __name__ == "__main__":
    load()
