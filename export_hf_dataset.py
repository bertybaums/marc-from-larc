#!/usr/bin/env python3
"""Export marc.db to HuggingFace Datasets format (Parquet).

Creates a multi-config dataset:
  - tasks:          577 task metadata records
  - task_subsets:    subset classifications per model
  - descriptions:   1,500 figurative descriptions with domain tags
  - baseline:       3,952 baseline trial results
  - figurative:     8,225 figurative trial results

Behavioral columns (correct, cell_accuracy) are included.
Prompt text and raw API responses are excluded to keep the dataset
small and focused. Researchers needing full prompts can reconstruct
them from task + condition + examples using the ARC/LARC data, or
contact the authors.

Usage:
    python export_hf_dataset.py [--db marc.db] [--output hf_dataset/]
"""

import argparse
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def export_tasks(conn, out_dir):
    """Export tasks table."""
    rows = conn.execute("""
        SELECT task_id, arc_name, source, num_train,
               see_description, do_description, grid_description
        FROM tasks
        ORDER BY task_id
    """).fetchall()

    table = pa.table({
        "task_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "arc_name": pa.array([r[1] for r in rows]),
        "source": pa.array([r[2] for r in rows]),
        "num_train": pa.array([r[3] for r in rows], type=pa.int32()),
        "see_description": pa.array([r[4] for r in rows]),
        "do_description": pa.array([r[5] for r in rows]),
        "grid_description": pa.array([r[6] for r in rows]),
    })

    path = out_dir / "tasks" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  tasks: {len(rows)} rows -> {path}")


def export_task_subsets(conn, out_dir):
    """Export task_subsets table."""
    rows = conn.execute("""
        SELECT task_id, model_name, subset
        FROM task_subsets
        ORDER BY task_id, model_name
    """).fetchall()

    table = pa.table({
        "task_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "model_name": pa.array([r[1] for r in rows]),
        "subset": pa.array([r[2] for r in rows]),
    })

    path = out_dir / "task_subsets" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  task_subsets: {len(rows)} rows -> {path}")


def export_descriptions(conn, out_dir):
    """Export figurative_descriptions table."""
    rows = conn.execute("""
        SELECT fig_id, task_id, generator_model, variant,
               source_domain, metaphor,
               figurative_see, figurative_do, figurative_grid
        FROM figurative_descriptions
        ORDER BY task_id, variant
    """).fetchall()

    table = pa.table({
        "fig_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "generator_model": pa.array([r[2] for r in rows]),
        "variant": pa.array([r[3] for r in rows]),
        "source_domain": pa.array([r[4] for r in rows]),
        "metaphor": pa.array([r[5] for r in rows]),
        "figurative_see": pa.array([r[6] for r in rows]),
        "figurative_do": pa.array([r[7] for r in rows]),
        "figurative_grid": pa.array([r[8] for r in rows]),
    })

    path = out_dir / "descriptions" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  descriptions: {len(rows)} rows -> {path}")


def export_baseline(conn, out_dir):
    """Export baseline_trials table (behavioral results only)."""
    rows = conn.execute("""
        SELECT trial_id, task_id, model_name, condition,
               num_examples, correct, cell_accuracy
        FROM baseline_trials
        ORDER BY task_id, model_name, condition, num_examples
    """).fetchall()

    table = pa.table({
        "trial_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "model_name": pa.array([r[2] for r in rows]),
        "condition": pa.array([r[3] for r in rows]),
        "num_examples": pa.array([r[4] for r in rows], type=pa.int32()),
        "correct": pa.array([r[5] for r in rows], type=pa.int32()),
        "cell_accuracy": pa.array(
            [r[6] if r[6] is not None else float("nan") for r in rows],
            type=pa.float32()
        ),
    })

    path = out_dir / "baseline" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  baseline: {len(rows)} rows -> {path}")


def export_figurative(conn, out_dir):
    """Export figurative_trials table (behavioral results only)."""
    rows = conn.execute("""
        SELECT ft.trial_id, ft.fig_id, ft.task_id, ft.model_name,
               ft.num_examples, ft.correct, ft.cell_accuracy,
               fd.variant, fd.source_domain
        FROM figurative_trials ft
        JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
        ORDER BY ft.task_id, fd.variant, ft.num_examples
    """).fetchall()

    table = pa.table({
        "trial_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "fig_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[2] for r in rows], type=pa.int32()),
        "model_name": pa.array([r[3] for r in rows]),
        "num_examples": pa.array([r[4] for r in rows], type=pa.int32()),
        "correct": pa.array([r[5] for r in rows], type=pa.int32()),
        "cell_accuracy": pa.array(
            [r[6] if r[6] is not None else float("nan") for r in rows],
            type=pa.float32()
        ),
        "variant": pa.array([r[7] for r in rows]),
        "source_domain": pa.array([r[8] for r in rows]),
    })

    path = out_dir / "figurative" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  figurative: {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="marc.db")
    parser.add_argument("--output", default="hf_dataset")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    print(f"Exporting {db_path} -> {out_dir}/\n")

    export_tasks(conn, out_dir)
    export_task_subsets(conn, out_dir)
    export_descriptions(conn, out_dir)
    export_baseline(conn, out_dir)
    export_figurative(conn, out_dir)

    conn.close()
    print(f"\nDone. Dataset ready at {out_dir}/")


if __name__ == "__main__":
    main()
