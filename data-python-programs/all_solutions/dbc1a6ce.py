def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with 0s and scattered 1s
    2. Output is the same grid with 8s added between aligned pairs of 1s
    3. If a row contains exactly 2 ones, fill the space between them with 8s
    4. If a column contains exactly 2 ones, fill the space between them with 8s
    5. Isolated 1s (no pair in their row/column) remain unchanged
    6. All original 1s remain as 1s in the output

    Procedure:
    1. Create a copy of the input grid
    2. Find all positions with 1s and group them by row and column
    3. For each row with exactly 2 ones, fill 8s between them
    4. For each column with exactly 2 ones, fill 8s between them
    5. Return the modified grid
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find all 1s in the grid
    ones = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones.append((r, c))

    # Group 1s by row
    rows_with_ones = {}
    for r, c in ones:
        if r not in rows_with_ones:
            rows_with_ones[r] = []
        rows_with_ones[r].append(c)

    # Group 1s by column
    cols_with_ones = {}
    for r, c in ones:
        if c not in cols_with_ones:
            cols_with_ones[c] = []
        cols_with_ones[c].append(r)

    # Draw horizontal lines for rows with exactly 2 1s
    for row, col_positions in rows_with_ones.items():
        if len(col_positions) == 2:
            c1, c2 = sorted(col_positions)
            for c in range(c1 + 1, c2):
                result[row][c] = 8

    # Draw vertical lines for columns with exactly 2 1s
    for col, row_positions in cols_with_ones.items():
        if len(row_positions) == 2:
            r1, r2 = sorted(row_positions)
            for r in range(r1 + 1, r2):
                result[r][col] = 8

    return result
