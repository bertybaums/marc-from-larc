#!/usr/bin/env python3
"""Interactive HTML inspector for MARC puzzle experiments.

Two views:
  inspect  — Main inspector with tabbed figurative variants per puzzle
  compare  — Variant comparison view: same puzzle, different metaphors, aligned reasoning
"""

import json
from pathlib import Path

import click

from db import init_db, get_task, get_all_variants
from grids import grid_to_png_bytes, grid_to_base64_png, COLOR_RGB, COLOR_CODES
from tasks import resolve_arc_path


ALL_MODELS = ["gpt-oss-120b", "gpt-oss-20b"]


def _grid_to_img_tag(grid, label=None, border_color="#ccc"):
    if not grid:
        return f'<div class="grid-placeholder">No grid</div>'
    b64 = grid_to_base64_png(grid)
    label_html = f'<div class="grid-label">{label}</div>' if label else ''
    return f'{label_html}<img src="data:image/png;base64,{b64}" class="grid-img" style="border: 2px solid {border_color};">'


def _get_all_marc_tasks(conn, model_name):
    """Get all valid MARC tasks (original variant) for a model."""
    return conn.execute('''
        SELECT ft.task_id, MIN(ft.num_examples) as n,
               fd.metaphor, fd.figurative_see, fd.figurative_do, fd.figurative_grid,
               fd.fig_id, fd.generator_model, fd.variant
        FROM figurative_trials ft
        JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
        WHERE ft.model_name=?
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
    ''', (model_name, model_name, model_name)).fetchall()


def _get_trial_data(conn, task_id, model_name, condition):
    return conn.execute('''
        SELECT predicted_grid, reasoning, correct, cell_accuracy
        FROM baseline_trials
        WHERE task_id=? AND model_name=? AND condition=?
        ORDER BY num_examples DESC LIMIT 1
    ''', (task_id, model_name, condition)).fetchone()


def _get_figurative_trial(conn, fig_id, model_name, num_examples):
    return conn.execute('''
        SELECT predicted_grid, reasoning, correct, cell_accuracy
        FROM figurative_trials
        WHERE fig_id=? AND model_name=? AND num_examples=?
        LIMIT 1
    ''', (fig_id, model_name, num_examples)).fetchone()


def _get_variant_marc_status(conn, fig_id, model_name):
    """Check full MARC property for a variant + model.

    Returns (status, min_k) where status is one of:
      'valid'   — all 3 MARC conditions hold
      'na'      — examples alone suffice (MARC doesn't apply for this model)
      'fig_solves' — figurative alone succeeds (too precise)
      'fail'    — figurative + examples never succeeds
    """
    fig = conn.execute("SELECT task_id FROM figurative_descriptions WHERE fig_id=?",
                       (fig_id,)).fetchone()
    if not fig:
        return "fail", None
    task_id = fig["task_id"]

    # Condition 1: examples alone must fail
    eo = conn.execute(
        "SELECT correct FROM baseline_trials WHERE task_id=? AND model_name=? AND condition='examples_only' AND correct=1",
        (task_id, model_name),
    ).fetchone()
    if eo:
        return "na", None

    # Condition 2: figurative alone must fail
    fig_only = conn.execute(
        "SELECT correct FROM figurative_trials WHERE fig_id=? AND model_name=? AND num_examples=0",
        (fig_id, model_name),
    ).fetchone()
    if fig_only and fig_only["correct"] == 1:
        return "fig_solves", None

    # Condition 3: figurative + examples must succeed for some k
    fig_ex = conn.execute(
        "SELECT MIN(num_examples) as min_k FROM figurative_trials WHERE fig_id=? AND model_name=? AND num_examples>0 AND correct=1",
        (fig_id, model_name),
    ).fetchone()
    if fig_ex and fig_ex["min_k"] is not None:
        return "valid", fig_ex["min_k"]
    return "fail", None


def _marc_dots(conn, fig_id, models):
    """Return HTML for per-model MARC status dots (green/red/grey)."""
    dots = ""
    for model in models:
        status, _ = _get_variant_marc_status(conn, fig_id, model)
        if status == "valid":
            cls = "valid"
        elif status == "na":
            cls = "na"
        else:
            cls = "invalid"
        dots += f'<span class="dot {cls}" title="{model}"></span>'
    return dots


def _marc_badges(conn, fig_id, models):
    """Return HTML badges showing per-model MARC status."""
    badges = ""
    for model in models:
        status, min_k = _get_variant_marc_status(conn, fig_id, model)
        short = model.replace("gpt-oss-", "")
        if status == "valid":
            badges += f'<span class="status pass">{short}: n={min_k}</span> '
        elif status == "na":
            badges += f'<span class="status na">{short}: examples suffice</span> '
        elif status == "fig_solves":
            badges += f'<span class="status fail">{short}: fig alone solves</span> '
        else:
            badges += f'<span class="status fail">{short}: fail</span> '
    return badges


def _truncate(text, max_len):
    if not text:
        return ""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Encode to ASCII with XML entity escapes for any non-ASCII chars
    text = text.encode("ascii", "xmlcharrefreplace").decode("ascii")
    if len(text) > max_len:
        return text[:max_len] + f"\n\n... [{len(text) - max_len} chars truncated]"
    return text


def _build_puzzle_html(conn, marc_row, arc_task, task_row, all_test_models):
    """Build HTML for a single MARC puzzle with tabbed variants."""
    tid = marc_row["task_id"]
    n = marc_row["n"]

    train = arc_task["train"]
    test_input = arc_task["test"][0]["input"]
    test_output = arc_task["test"][0]["output"]

    source = task_row['source'] if 'source' in task_row.keys() else 'larc'
    source_badge = f' <span class="badge source-badge">marc-submission</span>' if source == 'marc-submission' else ''

    all_variants = get_all_variants(conn, tid)

    # Count valid alts per model
    valid_counts = {}
    total_alts = 0
    for v in all_variants:
        if v["variant"] == "original":
            continue
        total_alts += 1
        for model in all_test_models:
            status, _ = _get_variant_marc_status(conn, v["fig_id"], model)
            if status == "valid":
                valid_counts[model] = valid_counts.get(model, 0) + 1

    alt_summary = " | ".join(
        f'{m.replace("gpt-oss-", "")}: {valid_counts.get(m, 0)}/{total_alts}'
        for m in all_test_models
    )

    html = f'''
    <div class="puzzle" id="puzzle-{tid}">
      <div class="puzzle-header" onclick="togglePuzzle({tid})">
        <h2>
          <span class="toggle-icon" id="icon-{tid}">&#9654;</span>
          Task {tid}: "{marc_row['metaphor'][:60]}"{source_badge}
          <span class="badge">n={n}</span>
          <span class="badge variant-count">alts {alt_summary}</span>
          <span class="arc-name">{task_row['arc_name']}</span>
        </h2>
      </div>
      <div class="puzzle-body" id="body-{tid}" style="display:none;">
    '''

    # --- Training examples ---
    html += '<div class="section"><h3>Training Examples</h3><div class="examples-row">'
    for i, ex in enumerate(train):
        html += f'''
        <div class="example-pair">
          <div class="example-label">Example {i+1}</div>
          <div class="example-grids">
            {_grid_to_img_tag(ex["input"], "Input")}
            <div class="arrow">&#8594;</div>
            {_grid_to_img_tag(ex["output"], "Output")}
          </div>
        </div>'''
    html += '</div></div>'

    # --- Literal description (LARC only) ---
    if source == 'larc' and task_row['do_description']:
        html += f'''
        <div class="section">
          <h3>Literal Description</h3>
          <div class="description">
            <p><strong>See:</strong> {task_row['see_description']}</p>
            <p><strong>Do:</strong> {task_row['do_description']}</p>
            <p><strong>Grid:</strong> {task_row['grid_description']}</p>
          </div>
        </div>'''

    # --- Figurative description with variant tabs ---
    html += f'<div class="section"><h3>Figurative Descriptions ({len(all_variants)} variants)</h3>'

    # Tab bar
    html += f'<div class="variant-tabs" id="tabs-{tid}">'
    for vi, v in enumerate(all_variants):
        active = "active" if vi == 0 else ""
        domain_label = f' [{v["source_domain"]}]' if v["source_domain"] else ""
        dots = _marc_dots(conn, v["fig_id"], all_test_models)
        label = v["variant"]
        html += f'<button class="variant-tab {active}" onclick="switchVariant({tid},{vi})" id="tab-{tid}-{vi}">{dots}{label}{domain_label}</button>'
    html += '</div>'

    # Tab panels
    for vi, v in enumerate(all_variants):
        display = "block" if vi == 0 else "none"
        badges = _marc_badges(conn, v["fig_id"], all_test_models)

        html += f'<div class="variant-panel" id="panel-{tid}-{vi}" style="display:{display};">'
        html += f'<div class="description figurative">'
        html += f'<p class="metaphor-line">&#128161; <em>"{v["metaphor"]}"</em></p>'
        html += f'<p>{badges}</p>'
        if v["figurative_see"]:
            html += f'<p><strong>See:</strong> {v["figurative_see"]}</p>'
            html += f'<p><strong>Do:</strong> {v["figurative_do"]}</p>'
            html += f'<p><strong>Grid:</strong> {v["figurative_grid"]}</p>'
        html += '</div>'

        # Per-model trial results for this variant
        for model in all_test_models:
            status, min_k = _get_variant_marc_status(conn, v["fig_id"], model)
            short = model.replace("gpt-oss-", "")

            if status == "valid":
                # MARC valid: show the successful trial
                trial = _get_figurative_trial(conn, v["fig_id"], model, min_k)
                if trial:
                    pred_grid = json.loads(trial["predicted_grid"]) if trial["predicted_grid"] else None
                    reasoning = trial["reasoning"] or ""
                    html += f'''
                    <div class="variant-trial">
                      <div class="condition-detail"><strong>{short}</strong> &mdash; MARC valid, solved with {min_k} example{"s" if min_k > 1 else ""}</div>
                      <div class="pred-row">
                        {_grid_to_img_tag(test_input, "Input")}
                        <div class="arrow">&#8594;</div>
                        {_grid_to_img_tag(pred_grid, f"{short} Prediction", "#2a2")}
                        <div class="arrow">vs</div>
                        {_grid_to_img_tag(test_output, "Expected", "#2a2")}
                      </div>
                      <details class="reasoning">
                        <summary>{short} reasoning ({len(reasoning)} chars)</summary>
                        <pre>{_truncate(reasoning, 3000)}</pre>
                      </details>
                    </div>'''

            elif status == "na":
                # Examples alone suffice: show best figurative trial for reference
                best = conn.execute('''
                    SELECT predicted_grid, reasoning, cell_accuracy, num_examples, correct
                    FROM figurative_trials
                    WHERE fig_id=? AND model_name=? AND num_examples>0
                    ORDER BY correct DESC, cell_accuracy DESC LIMIT 1
                ''', (v["fig_id"], model)).fetchone()
                if best:
                    pred_grid = json.loads(best["predicted_grid"]) if best["predicted_grid"] else None
                    reasoning = best["reasoning"] or ""
                    correct = best["correct"]
                    border = "#9e9e9e"
                    result = "correct" if correct else f"cell_acc={best['cell_accuracy']:.2f}"
                    html += f'''
                    <div class="variant-trial variant-trial-na">
                      <div class="condition-detail"><strong>{short}</strong> &mdash; examples alone suffice (not MARC); figurative+{best["num_examples"]}ex: {result}</div>
                      <div class="pred-row">
                        {_grid_to_img_tag(pred_grid, f"{short} (not MARC)", border)}
                        <div class="arrow">vs</div>
                        {_grid_to_img_tag(test_output, "Expected", "#2a2")}
                      </div>
                      <details class="reasoning">
                        <summary>{short} reasoning ({len(reasoning)} chars)</summary>
                        <pre>{_truncate(reasoning, 3000)}</pre>
                      </details>
                    </div>'''
                else:
                    html += f'''
                    <div class="variant-trial variant-trial-na">
                      <div class="condition-detail"><strong>{short}</strong> &mdash; examples alone suffice (not MARC)</div>
                    </div>'''

            else:
                # Failed or figurative alone solves: show best attempt
                label = "figurative alone solves" if status == "fig_solves" else "failed"
                best = conn.execute('''
                    SELECT predicted_grid, reasoning, cell_accuracy, num_examples
                    FROM figurative_trials
                    WHERE fig_id=? AND model_name=? AND num_examples>0
                    ORDER BY cell_accuracy DESC LIMIT 1
                ''', (v["fig_id"], model)).fetchone()
                if best:
                    pred_grid = json.loads(best["predicted_grid"]) if best["predicted_grid"] else None
                    reasoning = best["reasoning"] or ""
                    acc = best["cell_accuracy"]
                    acc_str = f"{acc:.2f}" if acc is not None else "N/A"
                    html += f'''
                    <div class="variant-trial variant-trial-fail">
                      <div class="condition-detail"><strong>{short}</strong> &mdash; {label} (best cell_acc={acc_str} at k={best["num_examples"]})</div>
                      <div class="pred-row">
                        {_grid_to_img_tag(pred_grid, f"{short} Best Attempt", "#d44")}
                        <div class="arrow">vs</div>
                        {_grid_to_img_tag(test_output, "Expected", "#2a2")}
                      </div>
                      <details class="reasoning">
                        <summary>{short} reasoning ({len(reasoning)} chars)</summary>
                        <pre>{_truncate(reasoning, 3000)}</pre>
                      </details>
                    </div>'''

        html += '</div>'  # variant-panel

    html += '</div>'  # section

    # --- Test input + expected output ---
    html += f'''
    <div class="section">
      <h3>Test</h3>
      <div class="test-row">
        {_grid_to_img_tag(test_input, "Test Input")}
        <div class="arrow">&#8594;</div>
        {_grid_to_img_tag(test_output, "Expected Output", "#2a2")}
      </div>
    </div>'''

    # --- MARC Property Demonstration ---
    # Side-by-side: examples_only FAIL vs original figurative+examples PASS
    html += f'<div class="section"><h3>MARC Property (original clue)</h3>'
    html += '<div class="conditions-row">'
    for model in all_test_models:
        short = model.replace("gpt-oss-", "")
        orig_fig_id = marc_row["fig_id"]
        marc_status, orig_min_k = _get_variant_marc_status(conn, orig_fig_id, model)

        # Examples only
        trial_eo = _get_trial_data(conn, tid, model, "examples_only")
        pred_eo = json.loads(trial_eo["predicted_grid"]) if trial_eo and trial_eo["predicted_grid"] else None
        correct_eo = trial_eo["correct"] if trial_eo else None
        acc_eo = f'{trial_eo["cell_accuracy"]:.2f}' if trial_eo and trial_eo["cell_accuracy"] is not None else "N/A"
        status_eo = "FAIL" if not correct_eo else "PASS"
        class_eo = "fail" if not correct_eo else "pass"

        if marc_status == "na":
            # Examples alone suffice: not MARC for this model
            html += f'''
            <div class="condition-card" style="opacity:0.6; border-left: 3px solid #9e9e9e;">
              <div class="condition-title">{short} &mdash; <span class="status na">not MARC (examples suffice)</span></div>
              <div class="marc-demo-row">
                <div class="marc-demo-col">
                  <div class="condition-detail">Examples only <span class="status {class_eo}">{status_eo}</span> (acc: {acc_eo})</div>
                  {_grid_to_img_tag(pred_eo, "Prediction", "#2a2" if correct_eo else "#d44")}
                </div>
                <div class="marc-demo-col">
                  {_grid_to_img_tag(test_output, "Expected", "#2a2")}
                </div>
              </div>
            </div>'''
        else:
            # MARC applies: show examples_only FAIL vs figurative+examples
            if orig_min_k:
                trial_fig = _get_figurative_trial(conn, orig_fig_id, model, orig_min_k)
            else:
                trial_fig = None
            pred_fig = json.loads(trial_fig["predicted_grid"]) if trial_fig and trial_fig["predicted_grid"] else None
            correct_fig = trial_fig["correct"] if trial_fig else None
            acc_fig = f'{trial_fig["cell_accuracy"]:.2f}' if trial_fig and trial_fig["cell_accuracy"] is not None else "N/A"
            status_fig = "PASS" if correct_fig else ("FAIL" if trial_fig else "N/A")
            class_fig = "pass" if correct_fig else "fail"
            k_label = f'{orig_min_k} ex' if orig_min_k else "?"

            marc_label = f'<span class="status pass">MARC</span>' if marc_status == "valid" else f'<span class="status fail">not MARC</span>'

            html += f'''
            <div class="condition-card">
              <div class="condition-title">{short} &mdash; {marc_label}</div>
              <div class="marc-demo-row">
                <div class="marc-demo-col">
                  <div class="condition-detail">Examples only <span class="status {class_eo}">{status_eo}</span> (acc: {acc_eo})</div>
                  {_grid_to_img_tag(pred_eo, "Prediction", "#d44" if not correct_eo else "#2a2")}
                </div>
                <div class="marc-demo-col">
                  <div class="condition-detail">Figurative + {k_label} <span class="status {class_fig}">{status_fig}</span> (acc: {acc_fig})</div>
                  {_grid_to_img_tag(pred_fig, "Prediction", "#2a2" if correct_fig else "#d44")}
                </div>
                <div class="marc-demo-col">
                  {_grid_to_img_tag(test_output, "Expected", "#2a2")}
                </div>
              </div>
            </div>'''

    html += '</div></div>'

    html += '</div></div>'  # puzzle-body, puzzle
    return html


# ========== Comparison View ==========

def _build_comparison_html(conn, tid, models, arc_task, task_row):
    """Build HTML for variant comparison view — per-model results per variant."""
    train = arc_task["train"]
    test_input = arc_task["test"][0]["input"]
    test_output = arc_task["test"][0]["output"]

    all_variants = get_all_variants(conn, tid)

    html = f'<div class="compare-puzzle" id="compare-{tid}">'
    html += f'<h2>Task {tid}: {task_row["arc_name"]}</h2>'

    # Training examples
    html += '<div class="section"><h3>Training Examples</h3><div class="examples-row">'
    for i, ex in enumerate(train):
        html += f'''
        <div class="example-pair">
          <div class="example-grids">
            {_grid_to_img_tag(ex["input"], f"Ex{i+1} In")}
            <div class="arrow">&#8594;</div>
            {_grid_to_img_tag(ex["output"], f"Ex{i+1} Out")}
          </div>
        </div>'''
    html += '</div></div>'

    # Test
    html += f'''
    <div class="section">
      <div class="test-row">
        {_grid_to_img_tag(test_input, "Test Input")}
        <div class="arrow">&#8594;</div>
        {_grid_to_img_tag(test_output, "Expected Output", "#2a2")}
      </div>
    </div>'''

    # Each variant
    for v in all_variants:
        domain = v["source_domain"] or "&mdash;"
        variant_label = v["variant"]

        # Determine card border: valid if ANY model has MARC for this variant
        statuses = [_get_variant_marc_status(conn, v["fig_id"], m)[0] for m in models]
        any_valid = any(s == "valid" for s in statuses) or v["variant"] == "original"

        badges = _marc_badges(conn, v["fig_id"], models)

        html += f'''
        <div class="compare-card {'compare-valid' if any_valid else 'compare-invalid'}">
          <div class="compare-header">
            <strong>{variant_label}</strong>
            <span class="domain-tag">{domain}</span>
            {badges}
          </div>
          <div class="compare-metaphor">&#128161; <em>"{v['metaphor']}"</em></div>
          <div class="compare-models-row">'''

        # Per-model results side by side
        for model in models:
            status, min_k = _get_variant_marc_status(conn, v["fig_id"], model)
            short = model.replace("gpt-oss-", "")

            if status == "valid":
                trial = _get_figurative_trial(conn, v["fig_id"], model, min_k)
                pred_grid = json.loads(trial["predicted_grid"]) if trial and trial["predicted_grid"] else None
                reasoning = trial["reasoning"] if trial else ""
                border = "#2a2"
                status_label = f"MARC n={min_k}"
            elif status == "na":
                best = conn.execute('''
                    SELECT predicted_grid, reasoning, cell_accuracy, num_examples, correct
                    FROM figurative_trials
                    WHERE fig_id=? AND model_name=? AND num_examples>0
                    ORDER BY correct DESC, cell_accuracy DESC LIMIT 1
                ''', (v["fig_id"], model)).fetchone()
                pred_grid = json.loads(best["predicted_grid"]) if best and best["predicted_grid"] else None
                reasoning = best["reasoning"] if best else ""
                border = "#9e9e9e"
                status_label = "examples suffice (not MARC)"
            else:
                best = conn.execute('''
                    SELECT predicted_grid, reasoning, cell_accuracy, num_examples
                    FROM figurative_trials
                    WHERE fig_id=? AND model_name=? AND num_examples>0
                    ORDER BY cell_accuracy DESC LIMIT 1
                ''', (v["fig_id"], model)).fetchone()
                pred_grid = json.loads(best["predicted_grid"]) if best and best["predicted_grid"] else None
                reasoning = best["reasoning"] if best else ""
                border = "#d44"
                acc_str = f"{best['cell_accuracy']:.2f}" if best and best["cell_accuracy"] is not None else "N/A"
                label = "fig alone solves" if status == "fig_solves" else f"fail (best={acc_str})"
                status_label = label

            html += f'''
            <div class="compare-model-col">
              <div class="compare-model-header"><strong>{short}</strong> &mdash; {status_label}</div>
              <div class="pred-row">
                {_grid_to_img_tag(pred_grid, f"{short}", border)}
              </div>
              <details class="reasoning">
                <summary>{short} reasoning ({len(reasoning) if reasoning else 0} chars)</summary>
                <pre>{_truncate(reasoning, 5000)}</pre>
              </details>
            </div>'''

        html += '</div></div>'  # compare-models-row, compare-card

    html += '</div>'
    return html


# ========== CSS & JS ==========

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: auto; padding: 20px; background: #f5f5f5; }
h1 { color: #333; border-bottom: 2px solid #667; padding-bottom: 10px; }
h2 { color: #333; }
.puzzle { background: white; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.1); overflow: hidden; }
.puzzle-header { padding: 15px 20px; cursor: pointer; background: #fafafa; border-bottom: 1px solid #eee; }
.puzzle-header:hover { background: #f0f0f0; }
.puzzle-header h2 { margin: 0; font-size: 1.1em; color: #333; }
.toggle-icon { display: inline-block; width: 20px; transition: transform 0.2s; }
.toggle-icon.open { transform: rotate(90deg); }
.badge { background: #4a9eff; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; margin-left: 10px; }
.source-badge { background: #e67e22; }
.variant-count { background: #7c4dff; }
.arc-name { color: #999; font-size: 0.8em; margin-left: 10px; font-weight: normal; }
.puzzle-body { padding: 20px; }
.section { margin: 20px 0; padding: 15px; background: #fafafa; border-radius: 6px; }
.section h3 { margin: 0 0 10px 0; color: #555; font-size: 1em; }
.examples-row { display: flex; flex-wrap: wrap; gap: 20px; }
.example-pair { text-align: center; }
.example-label { font-weight: bold; color: #666; margin-bottom: 5px; font-size: 0.9em; }
.example-grids { display: flex; align-items: center; gap: 8px; }
.grid-img { image-rendering: pixelated; height: auto; max-height: 150px; border-radius: 3px; }
.grid-label { font-size: 0.75em; color: #888; margin-bottom: 2px; }
.grid-placeholder { width: 80px; height: 80px; background: #eee; display: flex; align-items: center; justify-content: center; color: #999; font-size: 0.8em; border-radius: 3px; }
.arrow { font-size: 1.5em; color: #999; padding: 0 5px; align-self: center; }
.description { padding: 10px; line-height: 1.6; }
.description p { margin: 4px 0; }
.figurative { background: #fff8e1; border-left: 3px solid #ffc107; }
.metaphor-line { font-size: 1.1em; margin-bottom: 8px !important; }
.test-row { display: flex; align-items: center; gap: 10px; }
.conditions-row { display: flex; gap: 20px; flex-wrap: wrap; }
.condition-detail { font-size: 0.85em; color: #666; margin-bottom: 10px; }
.status { padding: 2px 8px; border-radius: 10px; font-size: 0.8em; color: white; display: inline-block; margin-right: 4px; }
.status.fail { background: #e53935; }
.status.pass { background: #43a047; }
.status.pending { background: #999; }
.status.na { background: #9e9e9e; }
.reasoning { margin-top: 10px; }
.reasoning summary { cursor: pointer; color: #666; font-size: 0.85em; }
.reasoning pre { background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 0.8em;
                  white-space: pre-wrap; word-wrap: break-word; max-height: 400px; overflow-y: auto; }
.nav { position: sticky; top: 0; background: white; padding: 10px 20px; z-index: 100;
       box-shadow: 0 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-radius: 8px; }
.nav-links { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.nav-links a { padding: 3px 8px; background: #e3f2fd; border-radius: 4px; text-decoration: none;
               color: #1565c0; font-size: 0.85em; }
.nav-links a:hover { background: #bbdefb; }
.legend { margin: 8px 0; font-size: 0.85em; color: #555; }
.legend .dot { vertical-align: middle; }
.pred-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

/* Variant tabs */
.variant-tabs { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.variant-tab { padding: 5px 12px; border: 1px solid #ddd; border-radius: 4px 4px 0 0;
               background: #f5f5f5; cursor: pointer; font-size: 0.85em; border-bottom: none; }
.variant-tab.active { background: #fff8e1; border-color: #ffc107; font-weight: bold; }
.variant-tab:hover { background: #fff3cd; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 2px; }
.dot.valid { background: #43a047; }
.dot.invalid { background: #e53935; }
.dot.na { background: #9e9e9e; }
.variant-panel { border: 1px solid #ffc107; border-radius: 0 6px 6px 6px; padding: 10px; }
.variant-trial { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #ddd; }
.conditions-row { display: flex; gap: 20px; flex-wrap: wrap; }
.condition-card { flex: 1; min-width: 350px; border: 1px solid #ddd; border-radius: 6px; padding: 15px; background: white; }
.condition-title { font-weight: bold; margin-bottom: 8px; font-size: 1em; }
.marc-demo-row { display: flex; gap: 15px; align-items: flex-start; flex-wrap: wrap; }
.marc-demo-col { text-align: center; }
.variant-trial-fail { opacity: 0.7; }
.variant-trial-na { opacity: 0.6; border-left: 3px solid #9e9e9e; padding-left: 8px; }

/* Comparison view */
.compare-puzzle { background: white; border-radius: 8px; margin: 20px 0; padding: 20px;
                  box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
.compare-card { border: 1px solid #ddd; border-radius: 6px; padding: 15px; margin: 10px 0; }
.compare-valid { border-left: 4px solid #43a047; }
.compare-invalid { border-left: 4px solid #e53935; opacity: 0.7; }
.compare-header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.compare-metaphor { font-size: 1.05em; margin-bottom: 10px; padding: 5px; background: #fff8e1; border-radius: 4px; }
.compare-models-row { display: flex; gap: 20px; flex-wrap: wrap; }
.compare-model-col { flex: 1; min-width: 300px; border: 1px solid #eee; border-radius: 4px; padding: 10px; }
.compare-model-header { font-size: 0.9em; margin-bottom: 8px; color: #333; }
.domain-tag { background: #e8eaf6; color: #3949ab; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; }
"""

JS = """
function togglePuzzle(tid) {
    var body = document.getElementById('body-' + tid);
    var icon = document.getElementById('icon-' + tid);
    if (body.style.display === 'none') {
        body.style.display = 'block';
        icon.classList.add('open');
    } else {
        body.style.display = 'none';
        icon.classList.remove('open');
    }
}
function expandAll() {
    document.querySelectorAll('.puzzle-body').forEach(el => el.style.display = 'block');
    document.querySelectorAll('.toggle-icon').forEach(el => el.classList.add('open'));
}
function collapseAll() {
    document.querySelectorAll('.puzzle-body').forEach(el => el.style.display = 'none');
    document.querySelectorAll('.toggle-icon').forEach(el => el.classList.remove('open'));
}
function switchVariant(tid, idx) {
    var panels = document.querySelectorAll('[id^="panel-' + tid + '-"]');
    panels.forEach(p => p.style.display = 'none');
    var tabs = document.querySelectorAll('[id^="tab-' + tid + '-"]');
    tabs.forEach(t => t.classList.remove('active'));
    document.getElementById('panel-' + tid + '-' + idx).style.display = 'block';
    document.getElementById('tab-' + tid + '-' + idx).classList.add('active');
}
"""


@click.group()
def cli():
    """MARC Puzzle Inspector — interactive HTML views."""
    pass


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "base_model", default="gpt-oss-120b",
              help="Model whose MARC puzzles define the set")
@click.option("--output", "output_path", default="inspect.html")
@click.option("--limit", default=None, type=int)
def inspect(db_path, base_model, output_path, limit):
    """Generate interactive HTML inspector with tabbed figurative variants."""
    conn = init_db(db_path)
    marc_tasks = _get_all_marc_tasks(conn, base_model)

    if limit:
        marc_tasks = marc_tasks[:limit]

    click.echo(f"Building inspector for {len(marc_tasks)} MARC puzzles")
    click.echo(f"  Base model: {base_model}")
    click.echo(f"  Models: {', '.join(ALL_MODELS)}")

    # Summary stats
    total_variants = 0
    valid_by_model = {m: 0 for m in ALL_MODELS}
    for row in marc_tasks:
        variants = get_all_variants(conn, row["task_id"])
        for v in variants:
            if v["variant"] == "original":
                continue
            total_variants += 1
            for m in ALL_MODELS:
                status, _ = _get_variant_marc_status(conn, v["fig_id"], m)
                if status == "valid":
                    valid_by_model[m] += 1

    html_parts = [
        f'<html><head><meta charset="UTF-8"><title>MARC Puzzle Inspector</title>',
        f"<style>{CSS}</style>",
        f"<script>{JS}</script>",
        "</head><body>",
        "<h1>MARC Puzzle Inspector</h1>",
    ]

    # Navigation
    model_summary = " | ".join(
        f'{m.replace("gpt-oss-", "")}: {valid_by_model[m]}/{total_variants}'
        for m in ALL_MODELS
    )
    html_parts.append('<div class="nav">')
    n_larc = sum(1 for r in marc_tasks if get_task(conn, r["task_id"])["source"] == "larc")
    n_sub = len(marc_tasks) - n_larc
    html_parts.append(f'<p><strong>{len(marc_tasks)}</strong> MARC puzzles '
                      f'({n_larc} LARC + {n_sub} submissions, verified by {base_model}) | '
                      f'<strong>{total_variants}</strong> alternative clues | '
                      f'Valid alts: {model_summary}</p>')
    html_parts.append(f'<div class="legend">'
                      f'Tab dots (1st=120b, 2nd=20b): '
                      f'<span class="dot valid"></span> MARC valid &nbsp; '
                      f'<span class="dot invalid"></span> not MARC &nbsp; '
                      f'<span class="dot na"></span> examples suffice (N/A)'
                      f'</div>')
    html_parts.append('<button onclick="expandAll()">Expand All</button> ')
    html_parts.append('<button onclick="collapseAll()">Collapse All</button> ')
    html_parts.append(f'<a href="compare.html" style="margin-left:10px;">Variant Comparison View &rarr;</a>')
    html_parts.append('<div class="nav-links">')
    for row in marc_tasks:
        tid = row["task_id"]
        variants = get_all_variants(conn, tid)
        n_valid_120b = sum(1 for v in variants if v["variant"] != "original"
                           and _get_variant_marc_status(conn, v["fig_id"], "gpt-oss-120b")[0] == "valid")
        style = ""
        if n_valid_120b == 0:
            style = ' style="background:#ffcdd2; color:#b71c1c;"'
        elif n_valid_120b < 2:
            style = ' style="background:#fff9c4; color:#f57f17;"'
        html_parts.append(f'<a href="#puzzle-{tid}" '
                          f'onclick="document.getElementById(\'body-{tid}\').style.display=\'block\'"'
                          f'{style}>T{tid} ({n_valid_120b})</a>')
    html_parts.append('</div></div>')

    # Puzzles
    for i, marc_row in enumerate(marc_tasks):
        tid = marc_row["task_id"]
        task_row = get_task(conn, tid)

        arc_path = resolve_arc_path(None, task_row["arc_name"])
        if not arc_path:
            click.echo(f"  SKIP task {tid}: ARC file not found")
            continue

        with open(arc_path) as f:
            arc_task = json.load(f)

        click.echo(f"  [{i+1}/{len(marc_tasks)}] task {tid}: {marc_row['metaphor'][:50]}")
        html_parts.append(_build_puzzle_html(conn, marc_row, arc_task, task_row, ALL_MODELS))

    html_parts.append("</body></html>")

    Path(output_path).write_text("\n".join(html_parts))
    click.echo(f"\nInspector written to {output_path}")
    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "base_model", default="gpt-oss-120b",
              help="Model whose MARC puzzles define the set")
@click.option("--output", "output_path", default="compare.html")
@click.option("--limit", default=None, type=int)
def compare(db_path, base_model, output_path, limit):
    """Generate variant comparison view — all metaphors for each puzzle, per-model results."""
    conn = init_db(db_path)
    marc_tasks = _get_all_marc_tasks(conn, base_model)

    puzzles_with_alts = []
    for row in marc_tasks:
        variants = get_all_variants(conn, row["task_id"])
        if len(variants) > 1:
            puzzles_with_alts.append(row)

    if limit:
        puzzles_with_alts = puzzles_with_alts[:limit]

    click.echo(f"Building comparison view for {len(puzzles_with_alts)} puzzles")
    click.echo(f"  Models: {', '.join(ALL_MODELS)}")

    html_parts = [
        f'<html><head><meta charset="UTF-8"><title>MARC Variant Comparison</title>',
        f"<style>{CSS}</style>",
        "</head><body>",
        "<h1>MARC Variant Comparison</h1>",
        f'<p>Same puzzle, different metaphors &mdash; comparing reasoning across '
        f'{", ".join(ALL_MODELS)}. '
        f'<a href="inspect.html">&larr; Back to Inspector</a></p>',
        f'<div class="legend">'
        f'Badges: '
        f'<span class="status pass">120b: n=k</span> = MARC valid (solved with k examples) &nbsp;'
        f'<span class="status fail">20b: fail</span> = not MARC for that model'
        f'</div>',
    ]

    # Navigation
    html_parts.append('<div class="nav">')
    html_parts.append(f'<p>{len(puzzles_with_alts)} puzzles with variants</p>')
    html_parts.append('<div class="nav-links">')
    for row in puzzles_with_alts:
        html_parts.append(f'<a href="#compare-{row["task_id"]}">T{row["task_id"]}</a>')
    html_parts.append('</div></div>')

    for i, row in enumerate(puzzles_with_alts):
        tid = row["task_id"]
        task_row = get_task(conn, tid)
        arc_path = resolve_arc_path(None, task_row["arc_name"])
        if not arc_path:
            continue
        with open(arc_path) as f:
            arc_task = json.load(f)

        click.echo(f"  [{i+1}/{len(puzzles_with_alts)}] task {tid}")
        html_parts.append(_build_comparison_html(conn, tid, ALL_MODELS, arc_task, task_row))

    html_parts.append("</body></html>")

    Path(output_path).write_text("\n".join(html_parts))
    click.echo(f"\nComparison view written to {output_path}")
    conn.close()


if __name__ == "__main__":
    cli()
