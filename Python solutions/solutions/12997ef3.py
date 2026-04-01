def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing a shape made of 1s and scattered colored values (non-zero, non-one).
    2. Output is multiple copies of the 1-shape pattern, each filled with a different color.
    3. The colored copies are stacked vertically if the colors are arranged vertically in the input.
    4. The colored copies are stacked horizontally if the colors are arranged horizontally in the input.
    5. Colors are processed in order of their position (top to bottom, left to right).

    Procedure:
    1. Find the bounding box of all 1s in the grid
    2. Extract the pattern within that bounding box
    3. Find all colored values (non-zero, non-one) and their positions
    4. Determine if colors are arranged vertically or horizontally
    5. Create output by stacking the colored patterns accordingly
    """

    # Find all positions with 1s
    ones_positions = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                ones_positions.append((r, c))

    assert ones_positions, "No 1s found in grid"

    # Find bounding box of 1s
    min_r = min(pos[0] for pos in ones_positions)
    max_r = max(pos[0] for pos in ones_positions)
    min_c = min(pos[1] for pos in ones_positions)
    max_c = max(pos[1] for pos in ones_positions)

    # Extract the pattern
    pattern_height = max_r - min_r + 1
    pattern_width = max_c - min_c + 1
    pattern = [[0 for _ in range(pattern_width)] for _ in range(pattern_height)]

    for r, c in ones_positions:
        pattern[r - min_r][c - min_c] = 1

    # Find colored values and their positions
    colors_and_positions = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0 and grid[r][c] != 1:
                colors_and_positions.append((grid[r][c], r, c))

    assert colors_and_positions, "No colored values found"

    # Sort by position to determine stacking direction
    colors_and_positions.sort(key=lambda x: (x[1], x[2]))  # Sort by row, then column

    # Determine if arrangement is primarily vertical or horizontal
    if len(colors_and_positions) == 1:
        # Only one color, default to vertical
        stack_vertically = True
    else:
        # Check if colors are more spread vertically or horizontally
        rows = [row for _, row, col in colors_and_positions]
        cols = [col for _, row, col in colors_and_positions]

        row_spread = max(rows) - min(rows)
        col_spread = max(cols) - min(cols)

        stack_vertically = row_spread >= col_spread

    # Create output
    colors = [color for color, _, _ in colors_and_positions]

    if stack_vertically:
        # Stack patterns vertically
        result = []
        for color in colors:
            for r in range(pattern_height):
                row = []
                for c in range(pattern_width):
                    if pattern[r][c] == 1:
                        row.append(color)
                    else:
                        row.append(0)
                result.append(row)
    else:
        # Stack patterns horizontally
        result = []
        for r in range(pattern_height):
            row = []
            for color in colors:
                for c in range(pattern_width):
                    if pattern[r][c] == 1:
                        row.append(color)
                    else:
                        row.append(0)
            result.append(row)

    return result
