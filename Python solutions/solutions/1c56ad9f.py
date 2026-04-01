def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a colored rectangular shape on a background of zeros
    2. Output is the same grid with horizontal cyclic shifts applied to rows
    3. The shift pattern depends on color % 4 and repeats every 4 rows
    4. Color % 4 = 0: [+1, 0, -1, 0] (right, no shift, left, no shift)
    5. Color % 4 = 1: [-1, 0, +1, 0] (left, no shift, right, no shift)
    6. Color % 4 = 2 or 3: [0, -1, 0, +1] (no shift, left, no shift, right)

    Procedure:
    1. Find the bounding box of the non-zero region and determine the color
    2. Calculate shift pattern based on color % 4
    3. For each row in the bounding box, determine shift from pattern using row index % 4
    4. Apply the shift to all non-zero values in that row
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find bounding box and determine color
    min_row = max_row = min_col = max_col = None
    color = None

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                if color is None:
                    color = grid[r][c]
                if min_row is None:
                    min_row = max_row = r
                    min_col = max_col = c
                else:
                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

    if min_row is None:  # No colored cells
        return result

    # Determine shift pattern based on color % 4
    color_mod = color % 4
    if color_mod == 0:
        shift_pattern = [1, 0, -1, 0]
    elif color_mod == 1:
        shift_pattern = [-1, 0, 1, 0]
    else:  # color_mod in [2, 3]
        shift_pattern = [0, -1, 0, 1]

    # Apply shifts to each row in the pattern
    for r in range(min_row, max_row + 1):
        pattern_row_index = r - min_row
        shift = shift_pattern[pattern_row_index % 4]

        if shift != 0:
            # Clear the row first
            for c in range(cols):
                if grid[r][c] != 0:
                    result[r][c] = 0

            # Apply shift
            for c in range(cols):
                if grid[r][c] != 0:
                    new_c = c + shift
                    if 0 <= new_c < cols:
                        result[r][new_c] = grid[r][c]

    return result
