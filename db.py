"""SQLite helpers for the MARC-from-LARC pipeline."""

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(path="marc.db"):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


# --- tasks ---

def upsert_task(conn, task_id, arc_name, description_id, see, do, grid, num_train,
                source='larc'):
    conn.execute(
        """INSERT OR REPLACE INTO tasks
           (task_id, arc_name, description_id, see_description, do_description,
            grid_description, num_train, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (task_id, arc_name, description_id, see, do, grid, num_train, source),
    )
    conn.commit()


def get_task(conn, task_id):
    return conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()


def get_all_tasks(conn, source=None):
    if source:
        return conn.execute(
            "SELECT * FROM tasks WHERE source=? ORDER BY task_id", (source,)
        ).fetchall()
    return conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()


# --- baseline trials ---

def insert_baseline_trial(conn, task_id, model_name, condition, num_examples,
                          repeat_num, prompt_text):
    """Insert a baseline trial row. Returns trial_id (existing or new)."""
    conn.execute(
        """INSERT OR IGNORE INTO baseline_trials
           (task_id, model_name, condition, num_examples, repeat_num, prompt_text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, model_name, condition, num_examples, repeat_num, prompt_text),
    )
    conn.commit()
    row = conn.execute(
        """SELECT trial_id FROM baseline_trials
           WHERE task_id=? AND model_name=? AND condition=? AND num_examples=?
                 AND repeat_num=?""",
        (task_id, model_name, condition, num_examples, repeat_num),
    ).fetchone()
    return row["trial_id"]


def update_baseline_response(conn, trial_id, raw_response, response_text, latency_ms,
                             error=None):
    conn.execute(
        """UPDATE baseline_trials
           SET raw_response=?, response_text=?, latency_ms=?, error=?,
               response_at=datetime('now')
           WHERE trial_id=?""",
        (raw_response, response_text, latency_ms, error, trial_id),
    )
    conn.commit()


def update_baseline_evaluation(conn, trial_id, predicted_grid, reasoning, correct,
                               cell_accuracy):
    conn.execute(
        """UPDATE baseline_trials
           SET predicted_grid=?, reasoning=?, correct=?, cell_accuracy=?
           WHERE trial_id=?""",
        (predicted_grid, reasoning, correct, cell_accuracy, trial_id),
    )
    conn.commit()


def get_baseline_trial(conn, trial_id):
    return conn.execute(
        "SELECT * FROM baseline_trials WHERE trial_id=?", (trial_id,)
    ).fetchone()


def get_pending_baseline(conn, model_name=None, condition=None):
    """Trials inserted but not yet called."""
    sql = "SELECT * FROM baseline_trials WHERE response_text IS NULL AND error IS NULL"
    params = []
    if model_name:
        sql += " AND model_name=?"
        params.append(model_name)
    if condition:
        sql += " AND condition=?"
        params.append(condition)
    return conn.execute(sql, tuple(params)).fetchall()


# --- task subsets ---

def upsert_subset(conn, task_id, model_name, subset, min_examples=None):
    conn.execute(
        """INSERT OR REPLACE INTO task_subsets
           (task_id, model_name, subset, min_examples_with_language)
           VALUES (?, ?, ?, ?)""",
        (task_id, model_name, subset, min_examples),
    )
    conn.commit()


def get_subsets(conn, model_name=None, subset=None):
    sql = "SELECT * FROM task_subsets WHERE 1=1"
    params = []
    if model_name:
        sql += " AND model_name=?"
        params.append(model_name)
    if subset:
        sql += " AND subset=?"
        params.append(subset)
    return conn.execute(sql, tuple(params)).fetchall()


# --- figurative descriptions ---

def insert_figurative(conn, task_id, generator_model, metaphor, fig_see, fig_do,
                      fig_grid, generation_prompt=None, raw_generation=None,
                      variant='original', source_domain=None):
    conn.execute(
        """INSERT OR IGNORE INTO figurative_descriptions
           (task_id, generator_model, variant, source_domain, metaphor,
            figurative_see, figurative_do, figurative_grid,
            generation_prompt, raw_generation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (task_id, generator_model, variant, source_domain, metaphor, fig_see,
         fig_do, fig_grid, generation_prompt, raw_generation),
    )
    conn.commit()
    row = conn.execute(
        """SELECT fig_id FROM figurative_descriptions
           WHERE task_id=? AND generator_model=? AND variant=?""",
        (task_id, generator_model, variant),
    ).fetchone()
    return row["fig_id"] if row else None


def get_figurative(conn, task_id, generator_model=None, variant=None):
    if generator_model and variant:
        return conn.execute(
            """SELECT * FROM figurative_descriptions
               WHERE task_id=? AND generator_model=? AND variant=?""",
            (task_id, generator_model, variant),
        ).fetchone()
    if generator_model:
        return conn.execute(
            "SELECT * FROM figurative_descriptions WHERE task_id=? AND generator_model=?",
            (task_id, generator_model),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM figurative_descriptions WHERE task_id=?", (task_id,)
    ).fetchall()


def get_all_variants(conn, task_id):
    """Get all figurative description variants for a task, original first."""
    return conn.execute(
        """SELECT * FROM figurative_descriptions WHERE task_id=?
           ORDER BY CASE WHEN variant='original' THEN 0 ELSE 1 END, variant""",
        (task_id,),
    ).fetchall()


# --- figurative trials ---

def insert_figurative_trial(conn, fig_id, task_id, model_name, num_examples,
                            repeat_num, prompt_text):
    conn.execute(
        """INSERT OR IGNORE INTO figurative_trials
           (fig_id, task_id, model_name, num_examples, repeat_num, prompt_text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (fig_id, task_id, model_name, num_examples, repeat_num, prompt_text),
    )
    conn.commit()
    row = conn.execute(
        """SELECT trial_id FROM figurative_trials
           WHERE fig_id=? AND model_name=? AND num_examples=? AND repeat_num=?""",
        (fig_id, model_name, num_examples, repeat_num),
    ).fetchone()
    return row["trial_id"]


def update_figurative_response(conn, trial_id, raw_response, response_text, latency_ms,
                               error=None):
    conn.execute(
        """UPDATE figurative_trials
           SET raw_response=?, response_text=?, latency_ms=?, error=?,
               response_at=datetime('now')
           WHERE trial_id=?""",
        (raw_response, response_text, latency_ms, error, trial_id),
    )
    conn.commit()


def update_figurative_evaluation(conn, trial_id, predicted_grid, reasoning, correct,
                                 cell_accuracy):
    conn.execute(
        """UPDATE figurative_trials
           SET predicted_grid=?, reasoning=?, correct=?, cell_accuracy=?
           WHERE trial_id=?""",
        (predicted_grid, reasoning, correct, cell_accuracy, trial_id),
    )
    conn.commit()


def get_pending_figurative(conn, model_name=None):
    sql = "SELECT * FROM figurative_trials WHERE response_text IS NULL AND error IS NULL"
    params = []
    if model_name:
        sql += " AND model_name=?"
        params.append(model_name)
    return conn.execute(sql, tuple(params)).fetchall()
