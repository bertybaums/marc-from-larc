#!/usr/bin/env python3
"""Phase 5: Analysis and reporting — Plotly HTML charts."""

import json
from pathlib import Path

import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db import init_db, get_all_tasks, get_subsets


def _query_baseline_stats(conn, model_name):
    """Get baseline success rates per condition."""
    rows = conn.execute(
        """SELECT condition,
                  COUNT(*) as total,
                  SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct_count,
                  AVG(cell_accuracy) as avg_cell_acc
           FROM baseline_trials
           WHERE model_name=? AND correct IS NOT NULL
           GROUP BY condition""",
        (model_name,),
    ).fetchall()
    return {r["condition"]: dict(r) for r in rows}


def _query_subset_counts(conn, model_name):
    """Get subset sizes."""
    rows = conn.execute(
        """SELECT subset, COUNT(*) as count
           FROM task_subsets WHERE model_name=?
           GROUP BY subset""",
        (model_name,),
    ).fetchall()
    return {r["subset"]: r["count"] for r in rows}


def _query_marc_stats(conn, model_name, generator_model):
    """Get MARC puzzle statistics."""
    # Valid MARC: examples_only fails, fig_only fails, fig+examples succeeds
    marc_tasks = conn.execute(
        """SELECT ft.task_id, MIN(ft.num_examples) as min_k, fd.metaphor
           FROM figurative_trials ft
           JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
           WHERE ft.model_name=? AND fd.generator_model=?
             AND ft.correct=1 AND ft.num_examples > 0
             AND ft.task_id NOT IN (
               SELECT task_id FROM baseline_trials
               WHERE model_name=? AND condition='examples_only' AND correct=1
             )
             AND ft.fig_id NOT IN (
               SELECT fig_id FROM figurative_trials
               WHERE model_name=? AND num_examples=0 AND correct=1
             )
           GROUP BY ft.task_id""",
        (model_name, generator_model, model_name, model_name),
    ).fetchall()
    return marc_tasks


def _query_mn_pairs(conn, model_name, generator_model):
    """Get (m, n) pairs: m = min examples with literal, n = min examples with figurative."""
    rows = conn.execute(
        """SELECT ts.task_id, ts.min_examples_with_language as m,
                  MIN(ft.num_examples) as n
           FROM task_subsets ts
           JOIN figurative_descriptions fd ON ts.task_id = fd.task_id
           JOIN figurative_trials ft ON fd.fig_id = ft.fig_id
           WHERE ts.model_name=? AND fd.generator_model=? AND ft.model_name=?
             AND ft.correct=1 AND ft.num_examples > 0
             AND ts.min_examples_with_language IS NOT NULL
           GROUP BY ts.task_id""",
        (model_name, generator_model, model_name),
    ).fetchall()
    return rows


@click.command()
@click.option("--db", "db_path", default="marc.db")
@click.option("--model", "model_names", multiple=True, required=True,
              help="Model(s) to include in analysis")
@click.option("--generator-model", "generator_model", default="claude-sonnet")
@click.option("--output", "output_path", default="report.html")
def analyze(db_path, model_names, generator_model, output_path):
    """Generate HTML analysis report with Plotly charts."""
    conn = init_db(db_path)

    figs = []

    # --- Chart 1: Baseline success rates per model × condition ---
    fig1 = go.Figure()
    conditions = ["examples_only", "language_only", "both"]
    for model_name in model_names:
        stats = _query_baseline_stats(conn, model_name)
        rates = []
        for cond in conditions:
            if cond in stats:
                s = stats[cond]
                rates.append(s["correct_count"] / s["total"] * 100 if s["total"] > 0 else 0)
            else:
                rates.append(0)
        fig1.add_trace(go.Bar(name=model_name, x=conditions, y=rates))

    fig1.update_layout(
        title="Baseline Success Rate by Condition",
        yaxis_title="Success Rate (%)",
        barmode="group",
    )
    figs.append(fig1)

    # --- Chart 2: Subset distribution per model ---
    subset_names = ["examples_sufficient", "language_sufficient", "both_required", "unsolvable"]
    fig2 = go.Figure()
    for model_name in model_names:
        counts = _query_subset_counts(conn, model_name)
        fig2.add_trace(go.Bar(
            name=model_name,
            x=subset_names,
            y=[counts.get(s, 0) for s in subset_names],
        ))

    fig2.update_layout(
        title="Task Subset Distribution",
        yaxis_title="Number of Tasks",
        barmode="group",
    )
    figs.append(fig2)

    # --- Chart 3: Auto-MARC yield ---
    marc_counts = {}
    for model_name in model_names:
        marc_tasks = _query_marc_stats(conn, model_name, generator_model)
        marc_counts[model_name] = len(marc_tasks)

    fig3 = go.Figure(go.Bar(
        x=list(marc_counts.keys()),
        y=list(marc_counts.values()),
        text=list(marc_counts.values()),
        textposition="auto",
    ))
    fig3.update_layout(
        title="Auto-MARC Puzzle Yield by Model",
        yaxis_title="Number of Valid MARC Puzzles",
    )
    figs.append(fig3)

    # --- Chart 4: (m, n) scatter — information gap ---
    fig4 = go.Figure()
    for model_name in model_names:
        mn_pairs = _query_mn_pairs(conn, model_name, generator_model)
        if mn_pairs:
            ms = [r["m"] for r in mn_pairs]
            ns = [r["n"] for r in mn_pairs]
            tids = [r["task_id"] for r in mn_pairs]
            fig4.add_trace(go.Scatter(
                x=ms, y=ns, mode="markers",
                name=model_name,
                text=[f"task {t}" for t in tids],
                hovertemplate="task %{text}<br>m=%{x} n=%{y}<extra></extra>",
            ))

    fig4.add_trace(go.Scatter(
        x=[0, 10], y=[0, 10], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="m = n (no info loss)",
        showlegend=True,
    ))
    fig4.update_layout(
        title="Information Gap: Literal (m) vs Figurative (n) Min Examples",
        xaxis_title="m (min examples with literal description)",
        yaxis_title="n (min examples with figurative description)",
    )
    figs.append(fig4)

    # --- Chart 5: Cell accuracy distribution for baseline ---
    fig5 = go.Figure()
    for model_name in model_names:
        for cond in conditions:
            accs = conn.execute(
                """SELECT cell_accuracy FROM baseline_trials
                   WHERE model_name=? AND condition=? AND cell_accuracy IS NOT NULL""",
                (model_name, cond),
            ).fetchall()
            if accs:
                fig5.add_trace(go.Box(
                    y=[r["cell_accuracy"] for r in accs],
                    name=f"{model_name} / {cond}",
                ))

    fig5.update_layout(
        title="Cell Accuracy Distribution by Model × Condition",
        yaxis_title="Cell Accuracy",
    )
    figs.append(fig5)

    # --- Build HTML report ---
    html_parts = [
        "<html><head>",
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        "<style>body { font-family: sans-serif; max-width: 1200px; margin: auto; padding: 20px; }"
        ".chart { margin: 30px 0; }</style>",
        "</head><body>",
        "<h1>MARC-from-LARC Analysis Report</h1>",
    ]

    # Summary table
    html_parts.append("<h2>Summary</h2><table border='1' cellpadding='5'>")
    html_parts.append("<tr><th>Model</th><th>Baseline Tasks</th>"
                      "<th>Subsets Classified</th><th>MARC Puzzles</th></tr>")
    for model_name in model_names:
        stats = _query_baseline_stats(conn, model_name)
        total_baseline = sum(s["total"] for s in stats.values()) if stats else 0
        subset_total = sum(_query_subset_counts(conn, model_name).values())
        marc_count = marc_counts.get(model_name, 0)
        html_parts.append(
            f"<tr><td>{model_name}</td><td>{total_baseline}</td>"
            f"<td>{subset_total}</td><td>{marc_count}</td></tr>"
        )
    html_parts.append("</table>")

    for i, fig in enumerate(figs):
        div_id = f"chart_{i}"
        html_parts.append(f'<div class="chart" id="{div_id}"></div>')
        html_parts.append(f"<script>Plotly.newPlot('{div_id}', "
                          f"{fig.to_json()}.data, {fig.to_json()}.layout);</script>")

    # MARC puzzle details
    html_parts.append("<h2>Valid MARC Puzzles</h2>")
    for model_name in model_names:
        marc_tasks = _query_marc_stats(conn, model_name, generator_model)
        html_parts.append(f"<h3>{model_name} ({len(marc_tasks)} puzzles)</h3>")
        if marc_tasks:
            html_parts.append("<table border='1' cellpadding='5'>")
            html_parts.append("<tr><th>Task ID</th><th>Min Examples (n)</th>"
                              "<th>Metaphor</th></tr>")
            for row in marc_tasks:
                html_parts.append(
                    f"<tr><td>{row['task_id']}</td><td>{row['min_k']}</td>"
                    f"<td>{row['metaphor']}</td></tr>"
                )
            html_parts.append("</table>")

    html_parts.append("</body></html>")

    Path(output_path).write_text("\n".join(html_parts))
    click.echo(f"Report written to {output_path}")
    conn.close()


if __name__ == "__main__":
    analyze()
