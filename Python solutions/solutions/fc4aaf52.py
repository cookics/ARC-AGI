def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 16x16 grid with background color 8 and a pattern made of two other colors
    2. The pattern is split horizontally at its vertical midpoint
    3. Bottom half stays in place but with colors swapped
    4. Top half shifts horizontally to a new position with colors swapped
    5. The two non-background colors swap with each other

    Procedure:
    1. Find the bounding box of non-background (non-8) cells
    2. Identify the two non-background colors
    3. Find the vertical midpoint of the pattern
    4. For the bottom half: keep in place, swap colors
    5. For the top half: shift to column (7 - original_left_col), swap colors
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find bounding box and non-background colors
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1
    colors = set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)
                colors.add(grid[r][c])

    # Get the two non-background colors
    colors_list = list(colors)
    if len(colors_list) == 2:
        color_a, color_b = colors_list[0], colors_list[1]
    else:
        # Handle edge case
        color_a, color_b = colors_list[0], colors_list[0]

    # Create color swap mapping
    color_map = {color_a: color_b, color_b: color_a, 8: 8}

    # Find midpoint
    mid_row = (min_row + max_row) / 2

    # Calculate shift for top half
    # The top half shifts such that its left edge moves to position (7 - min_col)
    # So shift = (7 - min_col) - min_col = 7 - 2*min_col
    shift = 7 - 2 * min_col

    # Create output grid (copy of input with all 8s)
    result = [[8] * cols for _ in range(rows)]

    # Process each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                swapped_color = color_map[grid[r][c]]

                if r < mid_row:
                    # Top half - shift right
                    new_c = c + shift
                    if 0 <= new_c < cols:
                        result[r][new_c] = swapped_color
                else:
                    # Bottom half - keep in place
                    result[r][c] = swapped_color

    return result
