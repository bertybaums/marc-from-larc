CREATE TABLE IF NOT EXISTS tasks (
    task_id          INTEGER PRIMARY KEY,  -- LARC 0-399, MARC submissions 1000+
    arc_name         TEXT NOT NULL UNIQUE,
    description_id   TEXT NOT NULL,        -- UUID of selected description
    see_description  TEXT NOT NULL,
    do_description   TEXT NOT NULL,
    grid_description TEXT NOT NULL,
    num_train        INTEGER NOT NULL,
    source           TEXT NOT NULL DEFAULT 'larc'  -- 'larc' or 'marc-submission'
);

CREATE TABLE IF NOT EXISTS baseline_trials (
    trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id        INTEGER NOT NULL REFERENCES tasks(task_id),
    model_name     TEXT NOT NULL,
    condition      TEXT NOT NULL,      -- 'examples_only','language_only','both'
    num_examples   INTEGER NOT NULL,   -- how many training examples shown
    repeat_num     INTEGER NOT NULL DEFAULT 1,
    prompt_text    TEXT NOT NULL,
    raw_response   TEXT,
    response_text  TEXT,
    response_at    TEXT,
    latency_ms     INTEGER,
    error          TEXT,
    predicted_grid TEXT,               -- JSON 2D array
    reasoning      TEXT,               -- model's chain-of-thought
    correct        INTEGER,            -- 1/0/NULL
    cell_accuracy  REAL,
    UNIQUE(task_id, model_name, condition, num_examples, repeat_num)
);

CREATE TABLE IF NOT EXISTS task_subsets (
    task_id                    INTEGER NOT NULL REFERENCES tasks(task_id),
    model_name                 TEXT NOT NULL,
    subset                     TEXT NOT NULL,  -- 'examples_sufficient','language_sufficient','both_required','unsolvable'
    min_examples_with_language INTEGER,        -- m value
    PRIMARY KEY (task_id, model_name)
);

CREATE TABLE IF NOT EXISTS figurative_descriptions (
    fig_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          INTEGER NOT NULL REFERENCES tasks(task_id),
    generator_model  TEXT NOT NULL,     -- model that created the figurative version
    variant          TEXT NOT NULL DEFAULT 'original',  -- 'original', 'alt-1', 'alt-2', ...
    source_domain    TEXT,              -- metaphor domain: 'biology', 'architecture', etc.
    metaphor         TEXT NOT NULL,     -- one-line metaphor concept
    figurative_see   TEXT NOT NULL,
    figurative_do    TEXT NOT NULL,
    figurative_grid  TEXT NOT NULL,
    generation_prompt TEXT,
    raw_generation   TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(task_id, generator_model, variant)
);

CREATE TABLE IF NOT EXISTS figurative_trials (
    trial_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    fig_id         INTEGER NOT NULL REFERENCES figurative_descriptions(fig_id),
    task_id        INTEGER NOT NULL REFERENCES tasks(task_id),
    model_name     TEXT NOT NULL,      -- model being tested
    num_examples   INTEGER NOT NULL,   -- 0 = figurative only
    repeat_num     INTEGER NOT NULL DEFAULT 1,
    prompt_text    TEXT NOT NULL,
    raw_response   TEXT,
    response_text  TEXT,
    response_at    TEXT,
    latency_ms     INTEGER,
    error          TEXT,
    predicted_grid TEXT,
    reasoning      TEXT,
    correct        INTEGER,
    cell_accuracy  REAL,
    UNIQUE(fig_id, model_name, num_examples, repeat_num)
);
