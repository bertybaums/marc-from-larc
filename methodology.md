# Supplemental Material: Methodology

## 1. Overview

We describe the methodology for constructing and validating MARC (Metaphor Abstraction and Reasoning Corpus) puzzles from two sources: (i) 400 tasks from the Language-complete ARC (LARC) dataset, each paired with human-authored literal descriptions, and (ii) 177 hand-crafted MARC submissions, each accompanied by a human-authored figurative clue. For each puzzle that satisfies the MARC property under an initial figurative clue, we generate domain-diverse alternative clues and verify that they, too, preserve the property. The result is a corpus of 64 MARC-verified puzzles, 250 alternative figurative clues drawn from over 60 source domains, and a complete set of model trial data---including reasoning traces---for two models at different scales.

## 2. Data Sources

### 2.1 LARC Tasks

The LARC dataset (Acquaviva et al., 2022) extends 400 ARC tasks (Chollet, 2019) with crowdsourced natural-language descriptions. Each task has a set of training input-output grid pairs and one held-out test pair. Each description consists of three components: a *see* description (what structures are visible in the input), a *do* description (the procedural transformation rule), and a *grid* description (how the output grid dimensions relate to the input). LARC tasks in our pipeline are assigned task IDs 0--399.

For each task, we selected the best available description using a priority ordering: verified descriptions first, then by annotator confidence, then by the number of successful programmatic builds. Training examples range from 2 to 10 per task (mean: 3.3).

### 2.2 MARC Submissions

We collected 177 hand-crafted MARC-style puzzles from human contributors. Each submission includes training input-output pairs, a held-out test pair (with output), and a single figurative clue (the `metaphor` field). These puzzles differ from LARC tasks in several respects: they were designed from the ground up as figurative puzzles, they have no literal see/do/grid descriptions, and their training sets are smaller (1--5 examples, mode: 2, mean: 2.1).

Submissions were normalized to standard ARC format (integer cell values 0--9, JSON with `train` and `test` keys) and assigned task IDs 1000--1176. Of the 177 files, 159 originally used string cell values (e.g., `"0"` rather than `0`), which were converted to integers during loading. One submission used a `title` field in place of `metaphor`, handled as a fallback.

### 2.3 Grid Representation

ARC grids use 10 colors indexed 0--9. For text-based prompts, we encode grids using single-character codes: `.` (black), `B` (blue), `R` (red), `G` (green), `Y` (yellow), `X` (grey), `M` (magenta), `O` (orange), `A` (azure), `W` (maroon). Each grid row is rendered as a space-separated string of these codes, one row per line.

## 3. Models

All experiments were conducted on the MindRouter infrastructure at the University of Idaho, which provides an OpenAI-compatible API endpoint. We tested three subject models:

| Model | Parameters | Modality | Temperature |
|-------|-----------|----------|-------------|
| gpt-oss-120b | 120B | text | 0.0 |
| gpt-oss-20b | 20B | text | 0.0 |
| qwen3.5-400b | 400B | text | 0.0 |

The gpt-oss models are reasoning models that place their output in a `reasoning_content` field rather than the standard `content` field, requiring a two-pass extraction approach (Section 4.1). All trials used temperature 0.0 to maximize reproducibility.

## 4. Evaluation Pipeline

### 4.1 Two-Pass LLM Protocol

Each trial uses a two-pass approach to separate reasoning from structured output extraction:

**Pass 1 (Reasoning).** The subject model receives the prompt and reasons freely. The prompt instructs the model to think step by step and conclude with a grid in a labeled `ANSWER:` block using the single-character color codes. For conditions that include language, the prompt also includes a Conceptual Digestion phase: the model is asked to identify the *cast* (roles of grid elements), *affordances* (what each element can do), and the *transformation rule* before predicting the output.

**Pass 2 (Extraction).** The reasoning from Pass 1 is fed to an extraction model, which converts the `ANSWER:` block into structured JSON (`{"reasoning": "...", "output_grid": [[...]]}`). For gpt-oss-120b and qwen3.5-400b, the same model handles both passes. For gpt-oss-20b, we use gpt-oss-120b for Pass 2, since the smaller model produces substantially higher parse failure rates when formatting its own output (~45% vs. ~18% with the larger extraction model).

Grid parsing employs multiple fallback strategies: JSON parsing, markdown code block extraction, `ANSWER:` block parsing, and direct grid-line detection. If parsing fails, the trial is retried up to 3 times before recording a parse error.

### 4.2 Experimental Conditions

We test each task under the following conditions, depending on the data source:

**LARC tasks (literal descriptions available):**
- **examples_only**: Training grid pairs + test input. No language.
- **language_only**: Literal see/do/grid description + test input. No training examples.
- **both**: Literal description + all training examples + test input.

**MARC submissions and figurative testing (both sources):**
- **figurative_only** (k=0): Figurative clue + test input. No training examples.
- **figurative + k examples** (k=1,...,num_train): Figurative clue + k training examples + test input.

For MARC submissions, the figurative prompt uses a unified format (single metaphor string) rather than the see/do/grid split used for LARC literal descriptions. The prompt dispatcher selects the appropriate format automatically based on whether the `figurative_see` field is populated.

### 4.3 Evaluation Metrics

**Exact match.** A prediction is correct if every cell matches the expected output grid.

**Cell accuracy.** The fraction of cells that match between predicted and expected grids. When grids have identical dimensions, this is simply the proportion of matching cells. When dimensions differ, the comparison pads the smaller grid with sentinel values, resulting in low but non-zero accuracy reflecting any partial overlap.

## 5. Phase 1: Baseline Collection

For each LARC task, we tested all three conditions (examples_only, language_only, both) with the full set of training examples, recording the prompt, raw model responses (both passes), reasoning trace, predicted grid, correctness, and cell accuracy. For MARC submissions, we tested only examples_only (no literal descriptions exist).

Baseline results for examples_only:

| | LARC (n=400) | Submissions (n=177) |
|---|:---:|:---:|
| gpt-oss-120b | 128 (32.0%) | 18 (10.2%) |
| gpt-oss-20b | 89 (22.2%) | 11 (6.2%) |
| qwen3.5-400b | 166 (41.5%) | --- |

The lower solve rate on submissions is expected: these puzzles were designed to require figurative understanding, so examples alone should be less informative.

## 6. Phase 2: Task Classification

LARC tasks were classified into four subsets based on baseline performance:

- **examples_sufficient**: Solved by examples alone.
- **language_sufficient**: Not solved by examples alone, but solved by literal description alone.
- **both_required**: Not solved by either alone, but solved by both together.
- **unsolvable**: Not solved under any baseline condition.

MARC submissions were classified into two categories: examples_sufficient (solved by examples alone) or unsolvable under the examples_only condition.

## 7. Phase 3: Figurative Description Generation

### 7.1 LARC Tasks

For LARC tasks classified as language_sufficient or both_required, we generated figurative descriptions using Claude (Anthropic) as the generator model. Each description consists of a one-line metaphor concept plus figurative versions of the see, do, and grid components. The generation prompt instructs the model to:

1. Assign imaginative roles to grid elements (colors, shapes, positions).
2. Describe the transformation through a metaphorical scenario.
3. Make the description evocative enough that seeing examples would trigger an "Aha!" moment.
4. Ensure the description is *not* precise enough to solve the puzzle without examples.

We generated 93 figurative descriptions across the eligible tasks. A quality check using Jaccard similarity between the literal and figurative *do* descriptions rejected candidates with similarity above 0.6, ensuring the figurative version was not merely a paraphrase.

### 7.2 MARC Submissions

The 177 submissions already include human-authored figurative clues. These were loaded directly with `generator_model='human'`.

## 8. Phase 4: MARC Property Verification

The MARC property is satisfied when three conditions hold simultaneously:

1. **Examples alone fail.** The model cannot solve the puzzle from training examples alone (examples_only condition).
2. **Figurative description alone fails.** The model cannot solve the puzzle from the figurative clue alone (figurative_only, k=0).
3. **Figurative description + examples succeeds.** The model solves the puzzle when given the figurative clue together with some number of training examples (figurative + k examples, for some k > 0).

For each task with a figurative description, we tested the figurative_only condition (k=0) and incrementally added training examples (k=1, 2, ..., num_train). The minimum k at which the model succeeds defines the *information gap*: the number of examples needed to bridge the gap between the figurative clue and the solution.

**Results (gpt-oss-120b, original clues):**
- LARC: 45/93 figurative descriptions satisfy the MARC property.
- Submissions: 19/177 submissions satisfy the MARC property.
- **Total: 64 MARC-verified puzzles.**

The lower yield for submissions reflects their heterogeneous quality: some metaphors are too vague to help even with examples (138 of 177 never succeeded), while others are too precise and solve the puzzle alone (2 of 177).

## 9. Phase 4b: Alternative Figurative Clues

### 9.1 Motivation

To support mechanistic interpretability research---specifically, investigating how different metaphorical framings of the same transformation are processed alongside grid examples---we generated multiple figurative clues for each MARC-verified puzzle. Each alternative is drawn from a different source domain (e.g., biology, warfare, cooking, music) to ensure the clues represent genuinely distinct metaphorical mappings rather than surface paraphrases.

### 9.2 Generation Procedure

Alternative clues were generated by Claude in three rounds, using progressively more structurally informative prompts:

**Round 1.** For each of the 64 MARC-verified puzzles, we generated 3 alternative metaphors (192 total). Each prompt included the original working metaphor, the literal description (when available), and instructions to produce clues from different source domains that encode the same structural transformation.

**Round 2.** For the 23 puzzles that had fewer than 2 MARC-valid alternatives after Round 1, we generated additional clues (34 total). The prompt was augmented with the failed alternatives and explicit instructions to embed more structural information.

**Round 3.** For the 18 puzzles still below the threshold, we generated a final batch (24 total). The prompt included detailed analysis of the literal `do_description` and instructions to map each specific operation (copy, mirror, count, fill, extend) to a concrete action in the chosen domain.

In total, we generated 250 alternative figurative clues across 63 distinct source domains.

### 9.3 Verification

Each alternative was tested identically to the original clues: figurative_only at k=0, then figurative + k examples for k=1,...,num_train. A clue preserves the MARC property if (1) examples alone still fail (inherited from the original verification), (2) the clue alone fails (k=0), and (3) the clue + some k > 0 examples succeeds.

### 9.4 Results

| | gpt-oss-120b | gpt-oss-20b |
|---|:---:|:---:|
| MARC-valid alternatives | 139/250 (55.6%) | 66/250 (26.4%) |
| Tasks with MARC property | 64/64 | 52/64 |
| Puzzles with >= 2 valid alts | 53/64 (82.8%) | --- |

Of the 64 tasks verified as MARC by gpt-oss-120b, 12 are solved by gpt-oss-20b under examples_only, meaning these tasks do not exhibit the MARC property for the smaller model. This is consistent with the scaling pattern: what requires figurative scaffolding at one scale may be solvable from examples alone at another.

Eleven puzzles resisted alternative figurative encoding across all three rounds and up to 7 attempts each. We term these "metaphorically narrow" puzzles: their transformations appear to admit only a very limited set of effective figurative framings. Three of these (tasks 1135, 1153, 1168) produced zero valid alternatives from any source domain.

Three alternatives across the corpus were too precise and solved the puzzle without examples (figurative_alone_succeeds), rendering them invalid despite successful figurative + examples performance.

The clear scaling effect---120b validates more alternatives than 20b---is consistent with larger models' greater capacity for integrating figurative language with example-based reasoning.

## 10. Data Storage

All experimental data is stored in a SQLite database (`marc.db`) with the following key tables:

- **tasks** (577 rows): Task metadata including `source` ('larc' or 'marc-submission') and task ID scheme (0--399 for LARC, 1000--1176 for submissions).
- **baseline_trials**: Per-task, per-model, per-condition trial results including prompt text, raw responses, reasoning traces, predicted grids, correctness, and cell accuracy.
- **figurative_descriptions**: Figurative clues with `variant` column ('original', 'alt-1', 'alt-2', ...) and `source_domain` column. Unique key: (task_id, generator_model, variant).
- **figurative_trials**: Per-variant, per-model, per-k trial results with the same structure as baseline_trials.

The variant storage design enables direct comparison of reasoning traces across different metaphorical framings of the same puzzle, facilitating mechanistic interpretability analysis.

## 11. Concurrency and Reproducibility

Batch evaluation jobs were executed with 8-way parallelism using Python's `ThreadPoolExecutor`, with each worker maintaining its own database connection. SQLite's WAL (Write-Ahead Logging) mode ensures safe concurrent writes. All API calls used temperature 0.0 for deterministic outputs, though we note that inference-time stochasticity in reasoning models may introduce minor variation across identical prompts.

## References

- Acquaviva, S., Pu, Y., Kryven, M., Vong, W. K., Sun, Z., & Tenenbaum, J. B. (2022). Communicating natural programs to humans and machines. *NeurIPS 2022*.
- Chollet, F. (2019). On the measure of intelligence. *arXiv:1911.01547*.
