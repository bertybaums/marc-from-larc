# Vision-Based MARC Testing: Results Report

## Overview

The text-based MARC pipeline tests whether figurative language combined with examples enables puzzle-solving that neither achieves alone. The vision pipeline asks the same question, but instead of sending text-rendered grids, it sends a single composite PNG image containing all training examples and the test input arranged in a vertical stack. The figurative clue remains in the prompt text: the model sees the grids visually and reads the metaphor linguistically.

Two vision-capable models were tested via MindRouter: `qwen3-vl-32b` (a dedicated vision-language model) and `qwen3.5-400b` (a large text model that also accepts image content blocks). Both use gpt-oss-120b for the extraction pass, consistent with the text pipeline's two-pass design.

## Baseline: Examples-Only with Composite Images

The first question is whether vision models can solve ARC puzzles from visual examples alone. The answer, in a nutshell: they mostly cannot.

| Model | Source | Correct | Accuracy | Parse Errors |
|-------|--------|---------|----------|-------------|
| qwen3-vl-32b | LARC (400) | 4 | 1.0% | 39 (10%) |
| qwen3-vl-32b | MARC sub (177) | 4 | 2.3% | 11 (6%) |
| qwen3.5-400b | LARC (400) | 10 | 2.5% | 161 (40%) |
| qwen3.5-400b | MARC sub (177) | 6 | 3.4% | 85 (48%) |

For comparison, the text pipeline's gpt-oss-120b solves roughly 15% of LARC tasks with examples alone. The vision models score an order of magnitude lower. The 400b model is somewhat stronger than the 32b (2.5% vs 1.0% on LARC), but both struggle considerably. Of note is the 400b model's high parse error rate (40-48%), suggesting it often produces responses the extraction pass cannot parse into a grid, despite using the same 120b extractor.

This low baseline is, paradoxically, favorable for MARC property testing: the fewer tasks solved by examples alone, the larger the pool of candidates where figurative language could make a difference.

### Subset Classification

| Model | Examples Sufficient | Examples Insufficient |
|-------|--------------------:|---------------------:|
| qwen3-vl-32b | 8 | 569 |
| qwen3.5-400b | 16 | 561 |

Nearly all tasks (98-99%) remain unsolved by visual examples alone.

## Figurative Testing

Each MARC candidate was tested at k=0 (figurative clue only, no examples) and k=1 through k=num_train (figurative clue plus increasing numbers of visual examples).

| Model | Source | Total Trials | Correct | k=0 | k>0 | Parse Errors |
|-------|--------|-------------|---------|-----|-----|-------------|
| qwen3-vl-32b | LARC | 405 | 8 | 2 | 6 | 18 (4%) |
| qwen3-vl-32b | MARC sub | 550 | 11 | 2 | 9 | 52 (9%) |
| qwen3.5-400b | LARC | 405 | 20 | 4 | 16 | 145 (36%) |
| qwen3.5-400b | MARC sub | 550 | 16 | 1 | 15 | 252 (46%) |

The 400b model again outperforms the 32b on raw correctness (36 vs 19 correct trials). But it also produces far more parse errors, especially on MARC submissions. The high parse error rate for 400b is a persistent issue across both baseline and figurative conditions, suggesting the model's verbosity interacts poorly with the two-pass extraction pipeline when processing images.

## MARC Property Verification

The MARC property requires all three conditions to hold simultaneously: (1) examples alone fail, (2) figurative clue alone fails, and (3) figurative clue with some number of examples succeeds. This is the critical test.

### qwen3-vl-32b

| Source | Valid MARC | Breakdown |
|--------|-----------|-----------|
| LARC | 2 | tasks 166, 390 |
| MARC sub | 0 | -- |
| **Total** | **2** | |

### qwen3.5-400b

| Source | Valid MARC | Breakdown |
|--------|-----------|-----------|
| LARC | 5 | tasks 166, 333, 346, 354, 390 |
| MARC sub | 4 | tasks 1048, 1074, 1134, 1158 |
| **Total** | **9** | |

### Cross-Model Overlap

The two tasks that satisfy the MARC property for both vision models are **task 166** (min_k=2 for both) and **task 390** (min_k=2 for 32b, min_k=1 for 400b). All of the 32b's valid MARC tasks are a subset of the 400b's: there are no puzzles where the smaller model succeeds but the larger one does not.

## Comparison with the Text Pipeline

The text pipeline using gpt-oss-120b identified 64 valid MARC puzzles across both LARC and MARC submissions. The vision pipeline (combining both models) finds 9. Five of these overlap with the text pipeline's results:

| Task | Text (120b) | Vision (32b) | Vision (400b) |
|------|:-----------:|:------------:|:-------------:|
| 166 | valid | valid (n=2) | valid (n=2) |
| 333 | valid | -- | valid (n=2) |
| 346 | valid | -- | valid (n=1) |
| 354 | valid | -- | valid (n=1) |
| 1134 | valid | -- | valid (n=1) |

Four puzzles satisfy the MARC property in vision but were not MARC-valid in the text pipeline:

- **Task 390** ("A roll call that skips the majority") -- valid for both vision models
- **Task 1048** ("Fall Vibes") -- valid for 400b only
- **Task 1074** ("Confusion cools into clarity.") -- valid for 400b only
- **Task 1158** ("Rationalists are yellow, Empiricists are blue") -- valid for 400b only

These vision-only MARC puzzles are potentially interesting: the figurative clue enables solution when the model *sees* the grids but not when it reads them as text. This suggests the metaphor may interact differently with visual versus textual representations of the same puzzle.

## Parse Error Analysis

A persistent concern across vision experiments is the high parse error rate, particularly for qwen3.5-400b:

| Model | Baseline Parse Errors | Figurative Parse Errors |
|-------|:---------------------:|:----------------------:|
| qwen3-vl-32b | 50 (9%) | 70 (7%) |
| qwen3.5-400b | 246 (43%) | 397 (42%) |

The 400b model produces unparseable responses nearly half the time. Since failed parses are scored as incorrect, the 400b's true capability may be somewhat higher than reported. But since the same extraction model (gpt-oss-120b) is used for both vision models, the difference is attributable to the 400b's response format rather than the extraction step. Admittedly, some of these parse errors could mask correct solutions, which would mean the true MARC-valid count for 400b may be modestly higher.

## Summary

The vision pipeline confirms that the MARC property generalizes beyond text-based grid representations: figurative language combined with visual examples can enable puzzle-solving that neither achieves alone. The effect is considerably rarer in vision (9 valid puzzles vs 64 in text), which is not surprising given that vision models perform substantially worse on ARC-style grid tasks overall. The 400b model, despite its parse error issues, finds more than four times as many MARC puzzles as the 32b (9 vs 2), and all 32b results are a strict subset.

There are a host of details that deserve further investigation. The four vision-only MARC puzzles merit close examination: why does the metaphor help when grids are seen but not when they are read? The high parse error rate for 400b may be masking additional valid cases. And alternative figurative clues, which already exist in the database and were copied to `marc_vision.db`, have not yet been tested in the vision modality -- that would allow the same kind of mechanistic interpretability comparisons (same puzzle, different metaphor, different reasoning path) that the text pipeline supports.

---

*Generated 2026-03-21. Database: `marc_vision.db`. Models tested: qwen3-vl-32b, qwen3.5-400b via MindRouter.*
