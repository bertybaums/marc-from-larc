"""LARC/ARC task loading, description selection, and database initialization."""

import csv
import json
from pathlib import Path

import click
import yaml

from db import init_db, upsert_task, get_all_tasks


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_dirs(config):
    """Return (arc_dir, larc_dir) Paths from config."""
    base = Path(__file__).parent
    return base / config["data"]["arc_dir"], base / config["data"]["larc_dir"]


def load_task_mapping(larc_dir):
    """Load LARC task_id → ARC task_name mapping from task.csv.

    Returns dict: {task_id: {"arc_name": str, "num_examples": int}}
    """
    csv_path = larc_dir / "dataset" / "summary" / "task.csv"
    mapping = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["task_id"])
            arc_name = row["task_name"].replace(".json", "")
            num_examples = int(row["number_of_examples"])
            mapping[tid] = {"arc_name": arc_name, "num_examples": num_examples}
    return mapping


def load_arc_task(arc_dir, arc_name):
    """Load an ARC task JSON. Returns {"train": [...], "test": [...]}."""
    path = arc_dir / f"{arc_name}.json"
    with open(path) as f:
        return json.load(f)


def load_larc_descriptions(larc_dir, task_id):
    """Load all descriptions for a LARC task.

    Returns dict: {description_id: {see_description, do_description, grid_description, ...}}
    """
    path = larc_dir / "dataset" / "tasks_json" / f"{task_id}.json"
    with open(path) as f:
        data = json.load(f)

    descriptions = {}
    descs = data.get("descriptions", data)
    for desc_id, desc in descs.items():
        if not isinstance(desc, dict):
            continue
        if "see_description" not in desc:
            continue
        descriptions[desc_id] = desc
    return descriptions


def select_best_description(descriptions):
    """Select the best description for a task.

    Priority: verified → highest confidence → most successful builds.
    Returns (description_id, description_dict) or (None, None).
    """
    if not descriptions:
        return None, None

    candidates = []
    for desc_id, desc in descriptions.items():
        verified = desc.get("succeeded_verification", False)
        confidence = desc.get("confidence", 0)
        # Count successful builds
        builds = desc.get("builds", {})
        successful_builds = sum(
            1 for b in builds.values()
            if isinstance(b, dict) and b.get("success", False)
        )
        candidates.append((desc_id, desc, verified, confidence, successful_builds))

    # Sort: verified first, then confidence desc, then successful builds desc
    candidates.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_id, best_desc = candidates[0][0], candidates[0][1]
    return best_id, best_desc


@click.group()
def cli():
    """MARC-from-LARC task management."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
def init(config_path, db_path):
    """Initialize the database with LARC tasks and best descriptions."""
    config = load_config(config_path)
    arc_dir, larc_dir = get_data_dirs(config)

    if not larc_dir.exists():
        click.echo(f"ERROR: LARC data not found at {larc_dir}")
        click.echo("Clone it: git clone https://github.com/samacqua/LARC.git data/larc")
        return

    if not arc_dir.exists():
        click.echo(f"ERROR: ARC data not found at {arc_dir}")
        click.echo("Clone it: git clone https://github.com/fchollet/ARC-AGI.git data/arc")
        return

    conn = init_db(db_path)
    mapping = load_task_mapping(larc_dir)

    loaded = 0
    skipped = 0
    for task_id in range(400):
        if task_id not in mapping:
            click.echo(f"  SKIP task {task_id}: not in task.csv")
            skipped += 1
            continue

        info = mapping[task_id]
        arc_name = info["arc_name"]

        # Verify ARC task file exists (check multiple possible locations)
        arc_path = None
        for subdir in ["data/training", "training", ""]:
            candidate = arc_dir / subdir / f"{arc_name}.json" if subdir else arc_dir / f"{arc_name}.json"
            if candidate.exists():
                arc_path = candidate
                break
        if arc_path is None:
            click.echo(f"  SKIP task {task_id}: ARC file not found for {arc_name}")
            skipped += 1
            continue

        # Load ARC task to get actual num_train
        with open(arc_path) as f:
            arc_data = json.load(f)
        num_train = len(arc_data["train"])

        # Load and select best description
        descriptions = load_larc_descriptions(larc_dir, task_id)
        desc_id, desc = select_best_description(descriptions)
        if desc_id is None:
            click.echo(f"  SKIP task {task_id}: no valid descriptions")
            skipped += 1
            continue

        upsert_task(
            conn, task_id, arc_name, desc_id,
            desc["see_description"],
            desc["do_description"],
            desc.get("grid_description", ""),
            num_train,
        )
        loaded += 1

    click.echo(f"Loaded {loaded} tasks, skipped {skipped}")
    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc.db")
def list_tasks(db_path):
    """List all tasks in the database."""
    conn = init_db(db_path)
    tasks = get_all_tasks(conn)
    for t in tasks:
        click.echo(
            f"  {t['task_id']:3d}  {t['arc_name']}  "
            f"train={t['num_train']}  "
            f"see={t['see_description'][:60]}..."
        )
    click.echo(f"Total: {len(tasks)} tasks")
    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.argument("task_id", type=int)
def show(db_path, task_id):
    """Show details of a specific task."""
    conn = init_db(db_path)
    from db import get_task
    t = get_task(conn, task_id)
    if not t:
        click.echo(f"Task {task_id} not found")
        return
    click.echo(f"Task {t['task_id']}: {t['arc_name']}")
    click.echo(f"Description: {t['description_id']}")
    click.echo(f"See: {t['see_description']}")
    click.echo(f"Do:  {t['do_description']}")
    click.echo(f"Grid: {t['grid_description']}")
    click.echo(f"Training examples: {t['num_train']}")
    conn.close()


def resolve_arc_path(config, arc_name):
    """Find the ARC task JSON file, checking multiple possible locations.

    Searches data/arc/ subdirectories first, then data/marc-submissions/.
    """
    base = Path(__file__).parent
    # Check ARC locations
    arc_dir = base / "data" / "arc"
    for subdir in ["data/training", "training", ""]:
        path = arc_dir / subdir / f"{arc_name}.json" if subdir else arc_dir / f"{arc_name}.json"
        if path.exists():
            return path
    # Check MARC submissions
    marc_path = base / "data" / "marc-submissions" / f"{arc_name}.json"
    if marc_path.exists():
        return marc_path
    return None


if __name__ == "__main__":
    cli()
