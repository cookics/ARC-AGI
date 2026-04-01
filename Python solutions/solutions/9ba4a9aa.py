def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The input is a large grid with a dominant background color.
    The output is a 3x3 block from the input that has a specific pattern:
    - 8 outer cells of one color (border)
    - 1 center cell of a different color
    This forms a cross-like pattern.

    Procedure:
    1. Find the background color (most frequent color)
    2. Scan the grid for 3x3 blocks that are not background
    3. For each 3x3 block, check if it has the border+center pattern
    4. Return the appropriate 3x3 block (appears to be the last valid one found)
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most frequent)
    color_count = {}
    for i in range(rows):
        for j in range(cols):
            color = grid[i][j]
            color_count[color] = color_count.get(color, 0) + 1

    background = max(color_count, key=color_count.get)

    # Find all valid 3x3 blocks
    valid_blocks = []

    for i in range(rows - 2):
        for j in range(cols - 2):
            # Extract 3x3 block
            block = []
            for r in range(i, i + 3):
                row = []
                for c in range(j, j + 3):
                    row.append(grid[r][c])
                block.append(row)

            # Check if this block has the border+center pattern and is not background
            if is_valid_block(block, background):
                valid_blocks.append((i, j, block))

    # Select the last valid block found (scanning left-to-right, top-to-bottom)
    if valid_blocks:
        return valid_blocks[-1][2]

    # Fallback - shouldn't happen with valid input
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def is_valid_block(block, background):
    """
    Check if a 3x3 block has the border+center pattern and is not background.
    Border+center means: all 8 outer cells same color, center cell different color.
    """
    if len(block) != 3 or len(block[0]) != 3:
        return False

    # Get border cells (8 outer cells)
    border_cells = [
        block[0][0],
        block[0][1],
        block[0][2],  # top row
        block[1][0],
        block[1][2],  # middle left and right
        block[2][0],
        block[2][1],
        block[2][2],  # bottom row
    ]

    center_cell = block[1][1]  # middle cell

    # Check if all border cells are the same color
    border_color = border_cells[0]
    if not all(cell == border_color for cell in border_cells):
        return False

    # Check if center is different from border
    if center_cell == border_color:
        return False

    # Check if this block is not just background
    if border_color == background:
        return False

    return True
