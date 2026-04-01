def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing 0s (empty cells), 8s (forming a rectangular block), and scattered colored numbers (non-0, non-8).
    2. Output is the same grid with colored numbers projected onto the boundary of the 8-block.
    3. Colored numbers that share the same row as the 8-block get projected to the left or right edge of the 8-block.
    4. Colored numbers that share the same column as the 8-block get projected to the top or bottom edge of the 8-block.
    5. If a colored number is to the left of the 8-block in the same row, it gets placed at the leftmost column of the 8-block.
    6. If a colored number is to the right of the 8-block in the same row, it gets placed at the rightmost column of the 8-block.
    7. If a colored number is above the 8-block in the same column, it gets placed at the topmost row of the 8-block.
    8. If a colored number is below the 8-block in the same column, it gets placed at the bottommost row of the 8-block.

    Procedure:
    1. Find the rectangular block of 8s (find min/max row and column indices)
    2. Find all colored numbers (non-0, non-8) and their positions
    3. For each colored number, check if it can be projected onto the 8-block boundary
    4. Place the projected numbers in the appropriate positions
    """

    # Make a copy of the grid
    result = [row[:] for row in grid]

    # Find the 8-block boundaries
    min_row, max_row = None, None
    min_col, max_col = None, None

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 8:
                if min_row is None or r < min_row:
                    min_row = r
                if max_row is None or r > max_row:
                    max_row = r
                if min_col is None or c < min_col:
                    min_col = c
                if max_col is None or c > max_col:
                    max_col = c

    # Find colored numbers and project them
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            val = grid[r][c]
            if val != 0 and val != 8:
                # Check if this number can be projected onto the 8-block

                # Same row projection
                if min_row <= r <= max_row:
                    if c < min_col:  # to the left
                        result[r][min_col] = val
                    elif c > max_col:  # to the right
                        result[r][max_col] = val

                # Same column projection
                if min_col <= c <= max_col:
                    if r < min_row:  # above
                        result[min_row][c] = val
                    elif r > max_row:  # below
                        result[max_row][c] = val

    return result
