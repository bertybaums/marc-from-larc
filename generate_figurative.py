#!/usr/bin/env python3
"""Phase 3: Generate figurative/metaphorical descriptions for candidate tasks."""

import json

import click

from db import (
    init_db, get_all_tasks, get_task, get_subsets, insert_figurative,
    get_figurative,
)
from models import call_llm_two_pass
from prompts import build_figurative_generation
from tasks import load_config


FIGURATIVE_EXTRACTION = """\
Below is reasoning about creating a metaphorical version of a grid puzzle instruction.

{reasoning}

Extract the final metaphorical description. Respond with ONLY this JSON, nothing else:
{{"metaphor":"<one-line metaphor concept>","see":"<figurative what you see>","do":"<figurative what to do>","grid":"<figurative grid details>"}}"""


def _build_figurative_extraction(reasoning):
    return [{"role": "user", "content": FIGURATIVE_EXTRACTION.format(reasoning=reasoning)}]


def _find_model_config(config, model_name):
    for m in config["models"]:
        if m["name"] == model_name:
            return m
    raise ValueError(f"Model '{model_name}' not found in config")


def _parse_figurative_response(text):
    """Parse the figurative generation response JSON.

    Returns (metaphor, see, do, grid) or raises ValueError.
    """
    text = text.strip()
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # Find JSON object
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    data = json.loads(text)

    metaphor = data.get("metaphor", "")
    see = data.get("see", "")
    do = data.get("do", "")
    grid = data.get("grid", "")

    if not metaphor or not see or not do:
        raise ValueError("Missing required fields in figurative response")

    return metaphor, see, do, grid


def _similarity_check(literal_do, figurative_do, threshold=0.6):
    """Basic string similarity check — reject if figurative is too close to literal.

    Uses word overlap ratio (Jaccard similarity on word sets).
    """
    lit_words = set(literal_do.lower().split())
    fig_words = set(figurative_do.lower().split())
    if not lit_words or not fig_words:
        return 0.0
    intersection = lit_words & fig_words
    union = lit_words | fig_words
    return len(intersection) / len(union)


@click.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc.db")
@click.option("--subject-model", "subject_model", required=True,
              help="Subject model whose subsets to use")
@click.option("--generator-model", "generator_model", default="claude-sonnet",
              help="Model to generate figurative descriptions")
@click.option("--limit", default=None, type=int)
@click.option("--similarity-threshold", default=0.6, type=float,
              help="Max word overlap ratio before rejecting as too literal")
@click.option("--dry-run", is_flag=True)
def generate(config_path, db_path, subject_model, generator_model, limit,
             similarity_threshold, dry_run):
    """Generate figurative descriptions for language_sufficient and both_required tasks."""
    config = load_config(config_path)
    gen_config = _find_model_config(config, generator_model)

    conn = init_db(db_path)
    subsets = get_subsets(conn, model_name=subject_model)
    candidates = [s for s in subsets
                  if s["subset"] in ("language_sufficient", "both_required")]

    if limit:
        candidates = candidates[:limit]

    click.echo(f"Generating figurative descriptions for {len(candidates)} tasks")
    click.echo(f"  Subject model: {subject_model}")
    click.echo(f"  Generator model: {generator_model}")

    if dry_run:
        for s in candidates:
            task = get_task(conn, s["task_id"])
            click.echo(f"  task={s['task_id']} ({s['subset']}) "
                        f"do={task['do_description'][:60]}...")
        return

    generated = 0
    skipped = 0
    rejected = 0
    errors = 0

    for i, s in enumerate(candidates, 1):
        task = get_task(conn, s["task_id"])
        tid = task["task_id"]

        # Skip if already generated
        existing = get_figurative(conn, tid, generator_model)
        if existing:
            click.echo(f"[{i}/{len(candidates)}] SKIP task={tid} (already generated)")
            skipped += 1
            continue

        messages = build_figurative_generation(task)
        generation_prompt = json.dumps(messages, ensure_ascii=False)

        click.echo(f"[{i}/{len(candidates)}] task={tid} ... ", nl=False)

        try:
            raw1, reasoning, raw2, extracted, latency_ms = call_llm_two_pass(
                gen_config, messages, _build_figurative_extraction
            )
            # Try extraction pass first, fall back to reasoning
            try:
                metaphor, fig_see, fig_do, fig_grid = _parse_figurative_response(extracted)
            except (json.JSONDecodeError, ValueError):
                metaphor, fig_see, fig_do, fig_grid = _parse_figurative_response(reasoning)
            raw = raw1

            # Quality check: similarity
            sim = _similarity_check(task["do_description"], fig_do, similarity_threshold)
            if sim > similarity_threshold:
                click.echo(f"REJECTED (similarity={sim:.2f})")
                rejected += 1
                continue

            insert_figurative(
                conn, tid, generator_model, metaphor, fig_see, fig_do, fig_grid,
                generation_prompt, raw,
            )
            click.echo(f"OK: \"{metaphor[:50]}\" (sim={sim:.2f}, {latency_ms}ms)")
            generated += 1

        except (json.JSONDecodeError, ValueError) as e:
            click.echo(f"PARSE ERROR: {e}")
            errors += 1
        except Exception as e:
            click.echo(f"ERROR: {e}")
            errors += 1

    click.echo(f"\nDone. generated={generated} skipped={skipped} "
               f"rejected={rejected} errors={errors}")
    conn.close()


if __name__ == "__main__":
    generate()
