#!/usr/bin/env python3
"""Bootstrap marc_vision.db from marc.db.

Creates marc_vision.db using schema.sql, then copies tasks and
figurative_descriptions (with explicit fig_id values) from marc.db.
Idempotent — safe to re-run.
"""

import sqlite3
from pathlib import Path

import click

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@click.command()
@click.option("--source-db", default="marc.db", help="Source database")
@click.option("--target-db", default="marc_vision.db", help="Target vision database")
def setup(source_db, target_db):
    """Create marc_vision.db and copy tasks + figurative_descriptions from marc.db."""
    # Create target DB with schema
    target = sqlite3.connect(target_db)
    target.row_factory = sqlite3.Row
    target.execute("PRAGMA journal_mode=WAL")
    target.executescript(SCHEMA_PATH.read_text())

    # Open source DB
    source = sqlite3.connect(source_db)
    source.row_factory = sqlite3.Row

    # Copy tasks
    tasks = source.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
    for t in tasks:
        target.execute(
            """INSERT OR IGNORE INTO tasks
               (task_id, arc_name, description_id, see_description, do_description,
                grid_description, num_train, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (t["task_id"], t["arc_name"], t["description_id"],
             t["see_description"], t["do_description"], t["grid_description"],
             t["num_train"], t["source"]),
        )
    target.commit()
    click.echo(f"Copied {len(tasks)} tasks")

    # Copy figurative_descriptions with explicit fig_id
    figs = source.execute(
        "SELECT * FROM figurative_descriptions ORDER BY fig_id"
    ).fetchall()
    for f in figs:
        target.execute(
            """INSERT OR IGNORE INTO figurative_descriptions
               (fig_id, task_id, generator_model, variant, source_domain,
                metaphor, figurative_see, figurative_do, figurative_grid,
                generation_prompt, raw_generation, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (f["fig_id"], f["task_id"], f["generator_model"], f["variant"],
             f["source_domain"], f["metaphor"], f["figurative_see"],
             f["figurative_do"], f["figurative_grid"], f["generation_prompt"],
             f["raw_generation"], f["created_at"]),
        )
    target.commit()
    click.echo(f"Copied {len(figs)} figurative descriptions")

    # Summary
    task_count = target.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    fig_count = target.execute("SELECT COUNT(*) FROM figurative_descriptions").fetchone()[0]
    click.echo(f"\nmarc_vision.db ready: {task_count} tasks, {fig_count} figurative descriptions")

    source.close()
    target.close()


if __name__ == "__main__":
    setup()
