"""Vision prompt builders for the composite-image MARC pipeline.

Unlike the existing _vision variants in prompts.py (which send multiple separate
image blocks, one per grid), these send ONE composite PNG containing all grids
arranged in a vertical stack.
"""

from grids import COLOR_KEY, compose_puzzle_image
from models import make_image_content_block
from prompts import REASONING_OUTPUT, CONCEPTUAL_DIGESTION


def build_vision_examples_only(arc_task, num_examples=None):
    """Build vision prompt for examples_only condition.

    Composite image shows k training examples + test input.
    """
    if num_examples is None:
        num_examples = len(arc_task["train"])

    image_bytes = compose_puzzle_image(arc_task, num_examples=num_examples)

    content_blocks = [
        {"type": "text", "text": (
            "You are solving a grid transformation puzzle. "
            "The image below shows training examples (input → output pairs) "
            "and a test input. Study the examples to discover the pattern, "
            "then predict the output for the test input.\n\n"
        )},
        make_image_content_block(image_bytes),
        {"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"},
    ]

    return [{"role": "user", "content": content_blocks}]


def build_vision_figurative_only(fig_row, arc_task):
    """Build vision prompt for figurative_only condition (k=0).

    Composite image shows only the test input.
    """
    image_bytes = compose_puzzle_image(arc_task, num_examples=0)

    # Build figurative description text
    fig_text = _format_figurative_text(fig_row)

    content_blocks = [
        {"type": "text", "text": (
            "You are solving a grid transformation puzzle. "
            "You are given a metaphorical description of the transformation "
            "and a test input grid image.\n\n"
            f"## Metaphorical Description\n\n{fig_text}\n\n"
            f"{CONCEPTUAL_DIGESTION}\n\n"
        )},
        make_image_content_block(image_bytes),
        {"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"},
    ]

    return [{"role": "user", "content": content_blocks}]


def build_vision_figurative_with_examples(fig_row, arc_task, num_examples):
    """Build vision prompt for figurative + k examples.

    Composite image shows k training examples + test input.
    """
    image_bytes = compose_puzzle_image(arc_task, num_examples=num_examples)

    fig_text = _format_figurative_text(fig_row)

    content_blocks = [
        {"type": "text", "text": (
            "You are solving a grid transformation puzzle. "
            "You are given a metaphorical description of the transformation, "
            "training examples (input → output pairs), and a test input.\n\n"
            f"## Metaphorical Description\n\n{fig_text}\n\n"
            f"{CONCEPTUAL_DIGESTION}\n\n"
        )},
        make_image_content_block(image_bytes),
        {"type": "text", "text": f"\n\n{COLOR_KEY}\n\n{REASONING_OUTPUT}"},
    ]

    return [{"role": "user", "content": content_blocks}]


def _format_figurative_text(fig_row):
    """Format figurative description text, handling unified vs see/do/grid split."""
    if not fig_row['figurative_see']:
        # Unified metaphor (MARC submissions, alternatives)
        return fig_row['metaphor']
    else:
        # See/do/grid split (LARC originals)
        return (
            f"What you see: {fig_row['figurative_see']}\n"
            f"What to do: {fig_row['figurative_do']}\n"
            f"Grid details: {fig_row['figurative_grid']}"
        )


def build_vision_prompt(condition, arc_task, fig_row=None, num_examples=None):
    """Dispatcher for vision prompts.

    Args:
        condition: 'examples_only', 'figurative_only', 'figurative_with_examples'
        arc_task: dict from ARC JSON
        fig_row: sqlite3.Row from figurative_descriptions (for figurative conditions)
        num_examples: number of training examples (None = all for examples_only)
    """
    if condition == "examples_only":
        return build_vision_examples_only(arc_task, num_examples)
    elif condition == "figurative_only":
        return build_vision_figurative_only(fig_row, arc_task)
    elif condition == "figurative_with_examples":
        return build_vision_figurative_with_examples(fig_row, arc_task, num_examples)
    else:
        raise ValueError(f"Unknown vision condition: {condition}")
