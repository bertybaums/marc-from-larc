"""Prompt templates for all MARC-from-LARC conditions."""

import json

from grids import (
    COLOR_KEY, format_examples_text, format_test_input_text,
    grid_to_text, grid_to_png_bytes,
)
from models import make_image_content_block

# --- Structured output instructions ---

# Pass 1: encourage reasoning, request the grid at the end
REASONING_OUTPUT = """\
Think step by step. After your reasoning, you MUST write out the complete predicted output grid using these single-character color codes:
""" + COLOR_KEY + """

At the very end of your response, write the complete output grid inside a clearly labeled block like this:
ANSWER:
. B R
G Y .
(one row per line, characters separated by spaces)"""

# Pass 2: extraction prompt — takes reasoning from pass 1 and extracts structured JSON
EXTRACTION_PROMPT = """\
Below is reasoning about a grid puzzle. It should contain an ANSWER block with the output grid.

<reasoning>
{reasoning}
</reasoning>

Extract the output grid from the ANSWER block above. Convert it to JSON format.
Do NOT re-analyze. Just extract and format the grid that was already produced.
Respond with ONLY this JSON object, nothing else:
{{"reasoning":"<one sentence summary>","output_grid":[[".","."],["B","R"]]}}

Color codes: .=black B=blue R=red G=green Y=yellow X=grey M=magenta O=orange A=azure W=maroon"""

# --- Conceptual Digestion phase (for conditions with language) ---

CONCEPTUAL_DIGESTION = """\
Before predicting the output, perform a Conceptual Digestion:
1. Identify the CAST: What roles do different grid elements play?
2. Identify the AFFORDANCES: What can each element do or have done to it?
3. Identify the TRANSFORMATION RULE: What is the precise rule that maps input to output?

Then apply the rule to the test input to predict the output grid."""

# --- Condition: examples_only ---

def build_examples_only(arc_task, num_examples=None):
    """Build messages for the examples_only condition (pure ARC).

    Provides training I/O pairs + test input. No language description.
    """
    train = arc_task["train"]
    if num_examples is not None:
        train = train[:num_examples]
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. Study the training examples to discover the pattern, then predict the output for the test input.

{COLOR_KEY}

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_examples_only_vision(arc_task, num_examples=None):
    """Build messages for examples_only with vision (PNG images)."""
    train = arc_task["train"]
    if num_examples is not None:
        train = train[:num_examples]
    test_input = arc_task["test"][0]["input"]

    content_blocks = [
        {"type": "text", "text": "You are solving a grid transformation puzzle. "
         "Study the training examples to discover the pattern, then predict the output "
         "for the test input.\n\n"},
    ]

    for i, ex in enumerate(train):
        content_blocks.append({"type": "text", "text": f"\nExample {i+1} Input:"})
        content_blocks.append(make_image_content_block(grid_to_png_bytes(ex["input"])))
        content_blocks.append({"type": "text", "text": f"\nExample {i+1} Output:"})
        content_blocks.append(make_image_content_block(grid_to_png_bytes(ex["output"])))

    content_blocks.append({"type": "text", "text": "\nTest Input:"})
    content_blocks.append(make_image_content_block(grid_to_png_bytes(test_input)))

    content_blocks.append({"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"})

    return [{"role": "user", "content": content_blocks}]


# --- Condition: language_only ---

def build_language_only(task_row, arc_task):
    """Build messages for the language_only condition (LARC).

    Provides literal description + test input only. No training examples.
    """
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a description of the transformation rule and a test input grid. Apply the rule to predict the output.

{COLOR_KEY}

## Transformation Description

What you see: {task_row['see_description']}
What to do: {task_row['do_description']}
Grid details: {task_row['grid_description']}

{CONCEPTUAL_DIGESTION}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_language_only_vision(task_row, arc_task):
    """Build messages for language_only with vision (PNG test input)."""
    test_input = arc_task["test"][0]["input"]

    content_blocks = [
        {"type": "text", "text": f"""\
You are solving a grid transformation puzzle. You are given a description of the transformation rule and a test input grid. Apply the rule to predict the output.

## Transformation Description

What you see: {task_row['see_description']}
What to do: {task_row['do_description']}
Grid details: {task_row['grid_description']}

{CONCEPTUAL_DIGESTION}

Test Input:"""},
        make_image_content_block(grid_to_png_bytes(test_input)),
        {"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"},
    ]

    return [{"role": "user", "content": content_blocks}]


# --- Condition: both (language + examples) ---

def build_both(task_row, arc_task, num_examples=None):
    """Build messages for the both condition (language + examples).

    Provides literal description + training I/O pairs + test input.
    """
    train = arc_task["train"]
    if num_examples is not None:
        train = train[:num_examples]
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a description of the transformation rule, training examples, and a test input grid.

{COLOR_KEY}

## Transformation Description

What you see: {task_row['see_description']}
What to do: {task_row['do_description']}
Grid details: {task_row['grid_description']}

{CONCEPTUAL_DIGESTION}

## Training Examples

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_both_vision(task_row, arc_task, num_examples=None):
    """Build messages for both condition with vision."""
    train = arc_task["train"]
    if num_examples is not None:
        train = train[:num_examples]
    test_input = arc_task["test"][0]["input"]

    content_blocks = [
        {"type": "text", "text": f"""\
You are solving a grid transformation puzzle. You are given a description of the transformation rule, training examples, and a test input grid.

## Transformation Description

What you see: {task_row['see_description']}
What to do: {task_row['do_description']}
Grid details: {task_row['grid_description']}

{CONCEPTUAL_DIGESTION}

## Training Examples
"""},
    ]

    for i, ex in enumerate(train):
        content_blocks.append({"type": "text", "text": f"\nExample {i+1} Input:"})
        content_blocks.append(make_image_content_block(grid_to_png_bytes(ex["input"])))
        content_blocks.append({"type": "text", "text": f"\nExample {i+1} Output:"})
        content_blocks.append(make_image_content_block(grid_to_png_bytes(ex["output"])))

    content_blocks.append({"type": "text", "text": "\nTest Input:"})
    content_blocks.append(make_image_content_block(grid_to_png_bytes(test_input)))
    content_blocks.append({"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"})

    return [{"role": "user", "content": content_blocks}]


# --- Figurative description generation prompt ---

def build_figurative_generation(task_row):
    """Build prompt for generating a figurative/metaphorical version of a literal description."""
    prompt = f"""\
You are transforming a literal puzzle instruction into a figurative/metaphorical version.

The literal instruction describes how to transform a grid puzzle:
- What you see: {task_row['see_description']}
- What to do: {task_row['do_description']}
- Grid details: {task_row['grid_description']}

Create a metaphorical version that:
1. Assigns imaginative roles to the grid elements (colors, shapes, positions)
2. Describes the transformation through a metaphorical scenario or narrative
3. Is evocative enough that seeing examples would make the solution click ("Aha!")
4. Is NOT precise enough to solve the puzzle without seeing any examples

Think of metaphors like: "The cat wants to play with the toy it cannot see" or "The floor is lava" or "DNA encodes the steps for replication."

The metaphorical description must still reference grid elements (colors, positions, regions) but through figurative language rather than literal procedural instructions.

Respond as JSON:
{{
  "metaphor": "<one-line metaphor concept>",
  "see": "<figurative version of what you see>",
  "do": "<figurative version of what to do>",
  "grid": "<figurative version of grid details>"
}}"""

    return [{"role": "user", "content": prompt}]


# --- Unified figurative prompt (single metaphor, no see/do/grid split) ---

def build_figurative_unified_only(fig_row, arc_task):
    """Build messages for a single-metaphor figurative description alone (k=0).

    Used for MARC submissions that have one metaphor string, not a see/do/grid split.
    """
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation and a test input grid.

{COLOR_KEY}

## Metaphorical Description

{fig_row['metaphor']}

{CONCEPTUAL_DIGESTION}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_figurative_unified_with_examples(fig_row, arc_task, num_examples):
    """Build messages for a single-metaphor figurative description + k training examples."""
    train = arc_task["train"][:num_examples]
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation, training examples, and a test input grid.

{COLOR_KEY}

## Metaphorical Description

{fig_row['metaphor']}

{CONCEPTUAL_DIGESTION}

## Training Examples

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


# --- Figurative testing prompts ---

def build_figurative_only(fig_row, arc_task):
    """Build messages for testing figurative description alone (k=0)."""
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation and a test input grid.

{COLOR_KEY}

## Metaphorical Description

What you see: {fig_row['figurative_see']}
What to do: {fig_row['figurative_do']}
Grid details: {fig_row['figurative_grid']}

{CONCEPTUAL_DIGESTION}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_figurative_with_examples(fig_row, arc_task, num_examples):
    """Build messages for figurative description + k training examples."""
    train = arc_task["train"][:num_examples]
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation, training examples, and a test input grid.

{COLOR_KEY}

## Metaphorical Description

What you see: {fig_row['figurative_see']}
What to do: {fig_row['figurative_do']}
Grid details: {fig_row['figurative_grid']}

{CONCEPTUAL_DIGESTION}

## Training Examples

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_figurative_only_vision(fig_row, arc_task):
    """Build vision messages for figurative description alone."""
    test_input = arc_task["test"][0]["input"]

    content_blocks = [
        {"type": "text", "text": f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation and a test input grid.

## Metaphorical Description

What you see: {fig_row['figurative_see']}
What to do: {fig_row['figurative_do']}
Grid details: {fig_row['figurative_grid']}

{CONCEPTUAL_DIGESTION}

Test Input:"""},
        make_image_content_block(grid_to_png_bytes(test_input)),
        {"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"},
    ]

    return [{"role": "user", "content": content_blocks}]


def build_figurative_with_examples_vision(fig_row, arc_task, num_examples):
    """Build vision messages for figurative + k examples."""
    train = arc_task["train"][:num_examples]
    test_input = arc_task["test"][0]["input"]

    content_blocks = [
        {"type": "text", "text": f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation, training examples, and a test input grid.

## Metaphorical Description

What you see: {fig_row['figurative_see']}
What to do: {fig_row['figurative_do']}
Grid details: {fig_row['figurative_grid']}

{CONCEPTUAL_DIGESTION}

## Training Examples
"""},
    ]

    for i, ex in enumerate(train):
        content_blocks.append({"type": "text", "text": f"\nExample {i+1} Input:"})
        content_blocks.append(make_image_content_block(grid_to_png_bytes(ex["input"])))
        content_blocks.append({"type": "text", "text": f"\nExample {i+1} Output:"})
        content_blocks.append(make_image_content_block(grid_to_png_bytes(ex["output"])))

    content_blocks.append({"type": "text", "text": "\nTest Input:"})
    content_blocks.append(make_image_content_block(grid_to_png_bytes(test_input)))
    content_blocks.append({"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"})

    return [{"role": "user", "content": content_blocks}]


# --- Dispatcher ---

def build_prompt(condition, task_row, arc_task, num_examples=None, fig_row=None,
                 vision=False):
    """Build prompt messages for any condition.

    Args:
        condition: 'examples_only', 'language_only', 'both',
                   'figurative_only', 'figurative_with_examples'
        task_row: sqlite3.Row from tasks table
        arc_task: dict from ARC JSON
        num_examples: number of training examples to include (None = all)
        fig_row: sqlite3.Row from figurative_descriptions table (for figurative conditions)
        vision: if True, use vision-compatible message format
    """
    if condition == "examples_only":
        if vision:
            return build_examples_only_vision(arc_task, num_examples)
        return build_examples_only(arc_task, num_examples)

    elif condition == "language_only":
        if vision:
            return build_language_only_vision(task_row, arc_task)
        return build_language_only(task_row, arc_task)

    elif condition == "both":
        if vision:
            return build_both_vision(task_row, arc_task, num_examples)
        return build_both(task_row, arc_task, num_examples)

    elif condition == "figurative_only":
        # Use unified prompt for single-metaphor descriptions (MARC submissions)
        if not fig_row['figurative_see']:
            return build_figurative_unified_only(fig_row, arc_task)
        if vision:
            return build_figurative_only_vision(fig_row, arc_task)
        return build_figurative_only(fig_row, arc_task)

    elif condition == "figurative_with_examples":
        if not fig_row['figurative_see']:
            return build_figurative_unified_with_examples(fig_row, arc_task, num_examples)
        if vision:
            return build_figurative_with_examples_vision(fig_row, arc_task, num_examples)
        return build_figurative_with_examples(fig_row, arc_task, num_examples)

    else:
        raise ValueError(f"Unknown condition: {condition}")


def build_extraction_messages(reasoning_text):
    """Build pass-2 extraction messages from pass-1 reasoning.

    Returns messages list that asks the model to extract the structured
    JSON answer from its own reasoning.
    """
    prompt = EXTRACTION_PROMPT.format(reasoning=reasoning_text)
    return [{"role": "user", "content": prompt}]
