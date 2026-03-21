"""Grid utilities: text ↔ grid conversion and PNG rendering for vision models."""

import base64
import io
import json
from typing import List

from PIL import Image, ImageDraw, ImageFont

# ARC color index → single-character code
COLOR_CODES = {
    0: ".",  # black
    1: "B",  # blue
    2: "R",  # red
    3: "G",  # green
    4: "Y",  # yellow
    5: "X",  # grey
    6: "M",  # magenta
    7: "O",  # orange
    8: "A",  # azure
    9: "W",  # maroon
}

CODE_TO_INDEX = {v: k for k, v in COLOR_CODES.items()}

COLOR_KEY = "Color key: .=black B=blue R=red G=green Y=yellow X=grey M=magenta O=orange A=azure W=maroon"

# ARC color index → RGB for PNG rendering
COLOR_RGB = {
    0: (0, 0, 0),         # black
    1: (0, 116, 217),     # blue
    2: (255, 65, 54),     # red
    3: (46, 204, 64),     # green
    4: (255, 220, 0),     # yellow
    5: (170, 170, 170),   # grey
    6: (240, 18, 190),    # magenta
    7: (255, 133, 27),    # orange
    8: (127, 219, 255),   # azure
    9: (128, 0, 0),       # maroon
}

CELL_SIZE = 20
BORDER_WIDTH = 1
BORDER_COLOR = (128, 128, 128)


def grid_to_text(grid: List[List[int]]) -> str:
    """Convert a 2D integer grid to a text representation with single-char color codes."""
    return "\n".join(" ".join(COLOR_CODES[cell] for cell in row) for row in grid)


def text_to_grid(text: str) -> List[List[int]]:
    """Convert text representation back to a 2D integer grid."""
    grid = []
    for line in text.strip().split("\n"):
        row = [CODE_TO_INDEX[c] for c in line.split()]
        grid.append(row)
    return grid


def grid_to_png_bytes(grid: List[List[int]]) -> bytes:
    """Render a 2D integer grid as a PNG image (20x20px cells, 1px grey border)."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    width = cols * CELL_SIZE + (cols + 1) * BORDER_WIDTH
    height = rows * CELL_SIZE + (rows + 1) * BORDER_WIDTH

    img = Image.new("RGB", (width, height), BORDER_COLOR)
    draw = ImageDraw.Draw(img)

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            x0 = BORDER_WIDTH + c * (CELL_SIZE + BORDER_WIDTH)
            y0 = BORDER_WIDTH + r * (CELL_SIZE + BORDER_WIDTH)
            x1 = x0 + CELL_SIZE - 1
            y1 = y0 + CELL_SIZE - 1
            draw.rectangle([x0, y0, x1, y1], fill=COLOR_RGB[cell])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def grid_to_base64_png(grid: List[List[int]]) -> str:
    """Render grid as base64-encoded PNG string."""
    return base64.b64encode(grid_to_png_bytes(grid)).decode("ascii")


def format_example_text(example: dict, idx: int) -> str:
    """Format a single training example as text.

    example: {"input": [[...]], "output": [[...]]}
    """
    lines = [f"Example {idx}:"]
    lines.append("Input:")
    lines.append(grid_to_text(example["input"]))
    lines.append("Output:")
    lines.append(grid_to_text(example["output"]))
    return "\n".join(lines)


def format_examples_text(examples: list, num_examples: int = None) -> str:
    """Format multiple training examples as text."""
    if num_examples is not None:
        examples = examples[:num_examples]
    return "\n\n".join(format_example_text(ex, i + 1) for i, ex in enumerate(examples))


def format_test_input_text(test_input: List[List[int]]) -> str:
    """Format test input grid as text."""
    return f"Test Input:\n{grid_to_text(test_input)}"


def compare_grids(predicted: List[List[int]], expected: List[List[int]]):
    """Compare two grids. Returns (correct: bool, cell_accuracy: float)."""
    if predicted == expected:
        return True, 1.0

    # Cell-level accuracy
    total = 0
    matching = 0
    max_rows = max(len(predicted), len(expected))
    for r in range(max_rows):
        pred_row = predicted[r] if r < len(predicted) else []
        exp_row = expected[r] if r < len(expected) else []
        max_cols = max(len(pred_row), len(exp_row))
        for c in range(max_cols):
            total += 1
            pred_val = pred_row[c] if c < len(pred_row) else -1
            exp_val = exp_row[c] if c < len(exp_row) else -2
            if pred_val == exp_val:
                matching += 1

    return False, matching / total if total > 0 else 0.0


def parse_response_grid(response_text: str):
    """Parse model response to extract predicted grid and reasoning.

    Tries multiple strategies:
    1. Parse as JSON with "reasoning" and "output_grid" fields
    2. Find JSON embedded in the response text
    3. Extract grid from an ANSWER: block in freeform text

    Returns (grid: list|None, reasoning: str|None, error: str|None).
    """
    if not response_text:
        return None, None, "Empty response"

    text = response_text.strip()

    # Strategy 1: try JSON parsing (possibly inside markdown code blocks)
    json_text = text
    if "```json" in json_text:
        try:
            start = json_text.index("```json") + 7
            end = json_text.index("```", start)
            json_text = json_text[start:end].strip()
        except ValueError:
            pass
    elif "```" in json_text:
        try:
            start = json_text.index("```") + 3
            end = json_text.index("```", start)
            json_text = json_text[start:end].strip()
        except ValueError:
            pass

    data = _try_parse_json(json_text)
    if data is None:
        # Try finding JSON object in the full text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            data = _try_parse_json(text[brace_start:brace_end + 1])

    if data is not None:
        reasoning = data.get("reasoning", "")
        output_grid = data.get("output_grid")
        if output_grid is not None:
            grid, err = _convert_grid(output_grid)
            if grid is not None:
                return grid, reasoning, None
            return None, reasoning, err
        return None, reasoning, "No output_grid in response JSON"

    # Strategy 2: try extracting grid from ANSWER: block
    grid, err = _extract_answer_block(text)
    if grid is not None:
        return grid, text, None

    return None, text, "Could not parse grid from response"


def _try_parse_json(text):
    """Try to parse text as JSON. Returns dict or None."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _convert_grid(output_grid):
    """Convert output_grid values to integers if needed. Returns (grid, error)."""
    if not output_grid or not output_grid[0]:
        return None, "Empty output_grid"
    if isinstance(output_grid[0][0], str):
        try:
            return [[CODE_TO_INDEX[c] for c in row] for row in output_grid], None
        except KeyError as e:
            return None, f"Unknown color code in grid: {e}"
    return output_grid, None


def _extract_answer_block(text):
    """Try to extract a grid from an ANSWER: block in freeform text.

    Looks for patterns like:
    ANSWER:
    . B R
    G Y .
    """
    import re

    # Look for ANSWER: followed by grid lines
    answer_match = re.search(r'ANSWER\s*:\s*\n((?:[.\w ]+\n?)+)', text, re.IGNORECASE)
    if not answer_match:
        # Try "Output:" or "Output grid:" as fallback markers
        answer_match = re.search(
            r'(?:Output grid|Output|Final grid|Final answer)\s*:\s*\n((?:[.\w ]+\n?)+)',
            text, re.IGNORECASE
        )

    if answer_match:
        grid_text = answer_match.group(1).strip()
        try:
            grid = text_to_grid(grid_text)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                return grid, None
        except (KeyError, ValueError, IndexError):
            pass

    # Try to extract the last sequence of numbered grid rows like:
    #   0: ". B R G"
    #   1: "Y . . ."
    numbered_rows = re.findall(
        r'^\s*\d+\s*:\s*"([.BRGXYMAOW ]+)"',
        text, re.MULTILINE
    )
    if len(numbered_rows) >= 2:
        # Take the last contiguous block of rows
        grid_text = "\n".join(numbered_rows)
        try:
            grid = text_to_grid(grid_text)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                return grid, None
        except (KeyError, ValueError, IndexError):
            pass

    # Try to find the last block of consecutive grid-like lines
    # (lines that are only color codes separated by spaces)
    grid_line_pattern = re.compile(r'^[.BRGXYMAOW](?:\s+[.BRGXYMAOW])+$')
    lines = text.split('\n')
    last_block_start = None
    last_block_end = None
    i = len(lines) - 1
    while i >= 0:
        if grid_line_pattern.match(lines[i].strip()):
            end = i
            while i >= 0 and grid_line_pattern.match(lines[i].strip()):
                i -= 1
            start = i + 1
            if end - start >= 1:  # at least 2 rows
                last_block_start = start
                last_block_end = end
                break
        i -= 1

    if last_block_start is not None:
        grid_text = "\n".join(lines[last_block_start:last_block_end + 1])
        try:
            grid = text_to_grid(grid_text)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                return grid, None
        except (KeyError, ValueError, IndexError):
            pass

    return None, "No grid found in response"


# --- Composite image builder for vision pipeline ---

ARROW_WIDTH = 40
LABEL_HEIGHT = 24
ROW_GAP = 16
PADDING = 12


def _get_font(size=14):
    """Try to load a clean font, fall back to PIL default."""
    for path in ["/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/SFNSMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _render_grid_to_image(grid):
    """Return a PIL Image for a single grid (no bytes conversion)."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    width = cols * CELL_SIZE + (cols + 1) * BORDER_WIDTH
    height = rows * CELL_SIZE + (rows + 1) * BORDER_WIDTH

    img = Image.new("RGB", (width, height), BORDER_COLOR)
    draw = ImageDraw.Draw(img)

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            x0 = BORDER_WIDTH + c * (CELL_SIZE + BORDER_WIDTH)
            y0 = BORDER_WIDTH + r * (CELL_SIZE + BORDER_WIDTH)
            x1 = x0 + CELL_SIZE - 1
            y1 = y0 + CELL_SIZE - 1
            draw.rectangle([x0, y0, x1, y1], fill=COLOR_RGB[cell])

    return img


def _draw_arrow(draw, x, y_center):
    """Draw a right-pointing arrow at (x, y_center)."""
    # Arrow shaft
    shaft_y = y_center
    shaft_left = x + 8
    shaft_right = x + ARROW_WIDTH - 12
    draw.line([(shaft_left, shaft_y), (shaft_right, shaft_y)], fill=(100, 100, 100), width=2)
    # Arrow head
    head_x = shaft_right
    head_size = 6
    draw.polygon([
        (head_x + head_size, shaft_y),
        (head_x - 2, shaft_y - head_size),
        (head_x - 2, shaft_y + head_size),
    ], fill=(100, 100, 100))


def compose_puzzle_image(arc_task, num_examples=0):
    """Compose a single PNG with training examples + test input.

    Layout (vertical stack):
        Example 1
        [input grid]  →  [output grid]

        Example 2
        [input grid]  →  [output grid]

        Test
        [input grid]  →  [ ? ]

    For num_examples=0: just test input with "Test Input" label.
    Returns PNG bytes.
    """
    font = _get_font()
    train = arc_task["train"][:num_examples] if num_examples > 0 else []
    test_input = arc_task["test"][0]["input"]

    # Pre-render all grid images
    rows_data = []  # list of (label, input_img, output_img_or_none)

    for i, ex in enumerate(train):
        in_img = _render_grid_to_image(ex["input"])
        out_img = _render_grid_to_image(ex["output"])
        rows_data.append((f"Example {i + 1}", in_img, out_img))

    test_in_img = _render_grid_to_image(test_input)
    rows_data.append(("Test", test_in_img, None))

    # Calculate canvas size
    max_width = 0
    total_height = PADDING

    row_layouts = []
    for label, in_img, out_img in rows_data:
        row_h = LABEL_HEIGHT
        pair_w = in_img.width
        pair_h = in_img.height
        if out_img is not None:
            pair_w += ARROW_WIDTH + out_img.width
            pair_h = max(in_img.height, out_img.height)
        else:
            # "?" placeholder same size as test input
            pair_w += ARROW_WIDTH + in_img.width
            pair_h = in_img.height

        row_total_h = row_h + pair_h
        row_layouts.append((pair_w, pair_h, row_total_h))
        max_width = max(max_width, pair_w)
        total_height += row_total_h + ROW_GAP

    total_height -= ROW_GAP  # no gap after last row
    total_height += PADDING
    canvas_width = max_width + 2 * PADDING

    # Draw
    canvas = Image.new("RGB", (canvas_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = PADDING

    for (label, in_img, out_img), (pair_w, pair_h, row_total_h) in zip(rows_data, row_layouts):
        # Label
        draw.text((PADDING, y), label, fill=(60, 60, 60), font=font)
        y += LABEL_HEIGHT

        # Input grid
        canvas.paste(in_img, (PADDING, y))

        # Arrow
        arrow_x = PADDING + in_img.width
        arrow_y_center = y + pair_h // 2
        _draw_arrow(draw, arrow_x, arrow_y_center)

        # Output grid or "?" placeholder
        out_x = PADDING + in_img.width + ARROW_WIDTH
        if out_img is not None:
            canvas.paste(out_img, (out_x, y))
        else:
            # Draw "?" placeholder
            placeholder_w = in_img.width
            placeholder_h = in_img.height
            draw.rectangle(
                [out_x, y, out_x + placeholder_w - 1, y + placeholder_h - 1],
                fill=(230, 230, 230), outline=(180, 180, 180), width=1,
            )
            # Center "?" text
            q_font = _get_font(size=24)
            bbox = draw.textbbox((0, 0), "?", font=q_font)
            q_w = bbox[2] - bbox[0]
            q_h = bbox[3] - bbox[1]
            draw.text(
                (out_x + (placeholder_w - q_w) // 2, y + (placeholder_h - q_h) // 2),
                "?", fill=(150, 150, 150), font=q_font,
            )

        y += pair_h + ROW_GAP

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
