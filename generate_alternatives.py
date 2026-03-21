#!/usr/bin/env python3
"""Alternative figurative clues for MARC-verified puzzles.

Generation is done by Claude Code Agent subagents (not API scripts).
This file provides:
  - list-targets: show MARC-verified puzzles needing alternatives
  - insert: insert a generated alternative into the DB
  - verify: test alternatives with gpt-oss-120b to confirm MARC property
  - report: summarize alternative clues and their verification status

Usage:
    # See what needs alternatives
    python generate_alternatives.py list-targets --model gpt-oss-120b --generator-model claude-agent

    # Insert a generated alternative
    python generate_alternatives.py insert --task-id 14 --generator-model claude-agent \
        --metaphor "The conductor assigns instruments" --source-domain "music"

    # Verify MARC property for unverified alternatives
    python generate_alternatives.py verify --model gpt-oss-120b

    # Report on alternatives
    python generate_alternatives.py report
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from db import (
    init_db, get_task, get_figurative, get_all_variants,
    insert_figurative, insert_figurative_trial,
    update_figurative_response, update_figurative_evaluation,
)
from grids import parse_response_grid, compare_grids
from models import call_llm_two_pass
from prompts import build_prompt, build_extraction_messages
from tasks import load_config
from utils import (
    find_model_config as _find_model_config,
    load_arc as _load_arc,
    serialize_prompt as _serialize_prompt,
)


def _get_marc_verified_tasks(conn, model_name, generator_model):
    """Get tasks with verified MARC property for a given model and generator."""
    return conn.execute('''
        SELECT ft.task_id, MIN(ft.num_examples) as min_k,
               fd.metaphor, fd.fig_id, fd.source_domain, fd.variant
        FROM figurative_trials ft
        JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
        WHERE ft.model_name=? AND fd.generator_model=?
          AND ft.correct=1 AND ft.num_examples > 0
          AND fd.variant = 'original'
          AND ft.task_id NOT IN (
            SELECT task_id FROM baseline_trials
            WHERE model_name=? AND condition='examples_only' AND correct=1
          )
          AND ft.fig_id NOT IN (
            SELECT fig_id FROM figurative_trials
            WHERE model_name=? AND num_examples=0 AND correct=1
          )
        GROUP BY ft.task_id
        ORDER BY ft.task_id
    ''', (model_name, generator_model, model_name, model_name)).fetchall()


@click.group()
def cli():
    """Alternative figurative clues for MARC puzzles."""
    pass


@cli.command("list-targets")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", default="gpt-oss-120b",
              help="Model used for MARC verification")
@click.option("--generator-model", "generator_model", required=True,
              help="Generator of original clues ('claude-agent' or 'human')")
@click.option("--num-alternatives", default=3, type=int,
              help="Target number of alternatives per puzzle")
def list_targets(db_path, model_name, generator_model, num_alternatives):
    """List MARC-verified puzzles that need alternative figurative clues."""
    conn = init_db(db_path)
    marc_tasks = _get_marc_verified_tasks(conn, model_name, generator_model)

    needs_alts = []
    for row in marc_tasks:
        existing = get_all_variants(conn, row["task_id"])
        alt_count = sum(1 for v in existing if v["variant"] != "original")
        remaining = num_alternatives - alt_count
        if remaining > 0:
            needs_alts.append((row, alt_count, remaining))

    click.echo(f"MARC-verified puzzles: {len(marc_tasks)} "
               f"({model_name}, generator={generator_model})")
    click.echo(f"Needing alternatives: {len(needs_alts)} "
               f"(target={num_alternatives} each)\n")

    for row, alt_count, remaining in needs_alts:
        task = get_task(conn, row["task_id"])
        source = task["source"] if "source" in task.keys() else "larc"
        # Show literal description for LARC tasks
        literal = ""
        if task["do_description"]:
            literal = (f"\n    See: {task['see_description'][:80]}"
                       f"\n    Do:  {task['do_description'][:80]}"
                       f"\n    Grid: {task['grid_description'][:80]}")
        existing_variants = get_all_variants(conn, row["task_id"])
        existing_metaphors = "\n    ".join(
            f"[{v['variant']}] \"{v['metaphor'][:70]}\"" for v in existing_variants
        )
        click.echo(f"task={row['task_id']} [{source}] min_k={row['min_k']} "
                    f"need={remaining} more")
        click.echo(f"    {existing_metaphors}")
        if literal:
            click.echo(f"    ---{literal}")
        click.echo()

    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--task-id", "task_id", required=True, type=int)
@click.option("--generator-model", "generator_model", required=True)
@click.option("--metaphor", required=True)
@click.option("--source-domain", "source_domain", required=True)
def insert(db_path, task_id, generator_model, metaphor, source_domain):
    """Insert a single alternative figurative clue into the database."""
    conn = init_db(db_path)

    task = get_task(conn, task_id)
    if not task:
        click.echo(f"ERROR: task {task_id} not found")
        return

    # Determine next variant number
    existing = get_all_variants(conn, task_id)
    alt_count = sum(1 for v in existing if v["variant"] != "original")
    variant_name = f"alt-{alt_count + 1}"

    fig_id = insert_figurative(
        conn, task_id, generator_model, metaphor,
        fig_see="", fig_do="", fig_grid="",
        variant=variant_name,
        source_domain=source_domain,
    )

    if fig_id:
        click.echo(f"Inserted: task={task_id} {variant_name} [{source_domain}] "
                    f"fig_id={fig_id}")
        click.echo(f"  \"{metaphor}\"")
    else:
        click.echo(f"ERROR: insert failed (may already exist)")

    conn.close()


@cli.command("insert-batch")
@click.option("--db", "db_path", default="marc.db")
@click.option("--json-file", "json_file", required=True,
              help="JSON file with alternatives: [{task_id, generator_model, metaphor, source_domain}, ...]")
def insert_batch(db_path, json_file):
    """Batch-insert alternative figurative clues from a JSON file."""
    conn = init_db(db_path)

    with open(json_file) as f:
        entries = json.load(f)

    inserted = 0
    for entry in entries:
        task_id = entry["task_id"]
        generator_model = entry["generator_model"]
        metaphor = entry["metaphor"]
        source_domain = entry["source_domain"]

        task = get_task(conn, task_id)
        if not task:
            click.echo(f"  SKIP task={task_id}: not found")
            continue

        existing = get_all_variants(conn, task_id)
        alt_count = sum(1 for v in existing if v["variant"] != "original")
        variant_name = f"alt-{alt_count + 1}"

        fig_id = insert_figurative(
            conn, task_id, generator_model, metaphor,
            fig_see="", fig_do="", fig_grid="",
            variant=variant_name,
            source_domain=source_domain,
        )

        if fig_id:
            click.echo(f"  task={task_id} {variant_name} [{source_domain}]: "
                        f"\"{metaphor[:50]}\"")
            inserted += 1

    click.echo(f"\nInserted {inserted}/{len(entries)} alternatives")
    conn.close()


from utils import get_extraction_model_config as _get_extraction_model_config


def _run_single_trial(model_config, config, db_path, fig_id, tid, k, variant, model_name):
    """Run a single verification trial. Thread-safe: each call opens its own DB connection."""
    conn = init_db(db_path)
    try:
        task = get_task(conn, tid)
        fig = conn.execute(
            "SELECT * FROM figurative_descriptions WHERE fig_id=?", (fig_id,)
        ).fetchone()

        arc_task = _load_arc(config, task["arc_name"])

        if k == 0:
            messages = build_prompt("figurative_only", task, arc_task, fig_row=fig)
        else:
            messages = build_prompt("figurative_with_examples", task, arc_task,
                                    num_examples=k, fig_row=fig)

        prompt_text = _serialize_prompt(messages)
        trial_id = insert_figurative_trial(conn, fig_id, tid, model_name, k, 1, prompt_text)

        # Skip if already done
        row = conn.execute(
            "SELECT response_text, error FROM figurative_trials WHERE trial_id=?",
            (trial_id,)
        ).fetchone()
        if row["response_text"] or row["error"]:
            return ("skip", tid, variant, k, "")

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
        return ("ok", tid, variant, k, f"{status} ({total_ms}ms){retries}")

    except Exception as e:
        try:
            update_figurative_response(conn, trial_id, None, None, 0, error=str(e))
        except Exception:
            pass
        return ("error", tid, variant, k, str(e))
    finally:
        conn.close()


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", default="gpt-oss-120b",
              help="Model to verify MARC property with")
@click.option("--concurrency", default=8, type=int,
              help="Number of parallel API calls (default: 8)")
@click.option("--limit", default=None, type=int)
@click.option("--dry-run", is_flag=True)
def verify(config_path, db_path, model_name, concurrency, limit, dry_run):
    """Verify MARC property for alternative figurative clues.

    Tests each unverified alternative at k=0 (should fail) and
    k=1,...,num_train (should succeed for some k).
    Runs API calls in parallel for throughput.
    """
    config = load_config(config_path)
    model_config = _find_model_config(config, model_name)

    conn = init_db(db_path)

    # Find alt variants without complete figurative trials
    alts = conn.execute('''
        SELECT fd.fig_id, fd.task_id, fd.metaphor, fd.variant, fd.source_domain
        FROM figurative_descriptions fd
        WHERE fd.variant != 'original'
        ORDER BY fd.task_id, fd.variant
    ''', ()).fetchall()

    if limit:
        alts = alts[:limit]

    # Build trial plan: k=0, then k=1,...,num_train for each alt
    trial_plan = []
    for alt in alts:
        task = get_task(conn, alt["task_id"])
        for k in range(0, task["num_train"] + 1):
            trial_plan.append((alt["fig_id"], alt["task_id"], k, alt["variant"]))

    click.echo(f"Verifying {len(alts)} alternative clues ({len(trial_plan)} trials)")
    click.echo(f"  Model: {model_name} | Concurrency: {concurrency}")

    if dry_run:
        for fig_id, tid, k, variant in trial_plan[:20]:
            click.echo(f"  task={tid} variant={variant} k={k}")
        if len(trial_plan) > 20:
            click.echo(f"  ... and {len(trial_plan) - 20} more")
        conn.close()
        return

    conn.close()  # Close main conn; workers open their own

    completed = 0
    skipped = 0
    errors = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_single_trial, model_config, config, db_path,
                fig_id, tid, k, variant, model_name
            ): (fig_id, tid, k, variant)
            for fig_id, tid, k, variant in trial_plan
        }

        for i, future in enumerate(as_completed(futures), 1):
            result_type, tid, variant, k, msg = future.result()
            with lock:
                if result_type == "skip":
                    skipped += 1
                elif result_type == "ok":
                    completed += 1
                    click.echo(f"[{i}/{len(trial_plan)}] task={tid} {variant} k={k} ... {msg}")
                else:
                    errors += 1
                    click.echo(f"[{i}/{len(trial_plan)}] task={tid} {variant} k={k} ... ERROR: {msg}")

                if i % 50 == 0:
                    click.echo(f"  --- progress: {completed} done, {skipped} skipped, {errors} errors ---")

    click.echo(f"\nDone. completed={completed} skipped={skipped} errors={errors}")


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_name", default="gpt-oss-120b")
def report(db_path, model_name):
    """Report on alternative figurative clues and their MARC verification status."""
    conn = init_db(db_path)

    alts = conn.execute('''
        SELECT fd.task_id, fd.variant, fd.source_domain, fd.metaphor, fd.fig_id
        FROM figurative_descriptions fd
        WHERE fd.variant != 'original'
        ORDER BY fd.task_id, fd.variant
    ''').fetchall()

    click.echo(f"\n=== Alternative Figurative Clues ===\n")
    click.echo(f"Total alternatives: {len(alts)}")

    marc_valid = 0
    marc_invalid = 0
    untested = 0

    for alt in alts:
        fig_id = alt["fig_id"]
        tid = alt["task_id"]

        trials = conn.execute(
            "SELECT num_examples, correct FROM figurative_trials WHERE fig_id=? AND model_name=?",
            (fig_id, model_name),
        ).fetchall()

        if not trials:
            untested += 1
            continue

        # Condition 1: examples alone must fail
        eo = conn.execute(
            "SELECT correct FROM baseline_trials WHERE task_id=? AND model_name=? AND condition='examples_only' AND correct=1",
            (tid, model_name),
        ).fetchone()
        if eo:
            marc_invalid += 1
            continue

        # Condition 2: figurative alone must fail
        fig_only = [t for t in trials if t["num_examples"] == 0]
        if fig_only and fig_only[0]["correct"] == 1:
            marc_invalid += 1
            continue

        # Condition 3: figurative + examples must succeed
        fig_ex = [t for t in trials if t["num_examples"] > 0 and t["correct"] == 1]
        if fig_ex:
            marc_valid += 1
        else:
            marc_invalid += 1

    click.echo(f"MARC-valid alternatives: {marc_valid}")
    click.echo(f"MARC-invalid alternatives: {marc_invalid}")
    click.echo(f"Untested alternatives: {untested}")

    # Domain distribution
    domains = {}
    for alt in alts:
        d = alt["source_domain"] or "untagged"
        domains[d] = domains.get(d, 0) + 1

    if domains:
        click.echo(f"\nSource domain distribution:")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            click.echo(f"  {domain}: {count}")

    # Per-task detail
    click.echo(f"\nPer-task breakdown:")
    by_task = {}
    for alt in alts:
        tid = alt["task_id"]
        if tid not in by_task:
            by_task[tid] = []
        by_task[tid].append(alt)

    for tid, task_alts in sorted(by_task.items()):
        # Check condition 1 once per task
        eo = conn.execute(
            "SELECT correct FROM baseline_trials WHERE task_id=? AND model_name=? AND condition='examples_only' AND correct=1",
            (tid, model_name),
        ).fetchone()
        task_eo_pass = eo is not None

        click.echo(f"  task={tid}: {len(task_alts)} alternatives" +
                    (" [examples_only passes]" if task_eo_pass else ""))
        for alt in task_alts:
            fig_id = alt["fig_id"]
            trials = conn.execute(
                "SELECT num_examples, correct FROM figurative_trials WHERE fig_id=? AND model_name=?",
                (fig_id, model_name),
            ).fetchall()
            if not trials:
                status = "untested"
            elif task_eo_pass:
                status = "INVALID (examples alone succeeds)"
            else:
                fig_only = [t for t in trials if t["num_examples"] == 0 and t["correct"] == 1]
                fig_ex = [t for t in trials if t["num_examples"] > 0 and t["correct"] == 1]
                if fig_only:
                    status = "INVALID (figurative alone succeeds)"
                elif fig_ex:
                    min_k = min(t["num_examples"] for t in fig_ex)
                    status = f"VALID (n={min_k})"
                else:
                    status = "INVALID (never succeeds)"
            domain = alt["source_domain"] or "?"
            click.echo(f"    {alt['variant']} [{domain}]: "
                        f"\"{alt['metaphor'][:50]}\" → {status}")

    conn.close()


if __name__ == "__main__":
    cli()
