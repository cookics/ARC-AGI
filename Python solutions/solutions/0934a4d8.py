def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid is 30x30 with a rectangular region of 8s marking corrupted values
    2. The grid has vertical (top-bottom) reflection symmetry around row 14.5
    3. For rows with 8s, look at their vertically symmetric rows
    4. If the symmetric row also has 8s, use the same column positions from non-8 rows

    Procedure:
    1. Find the bounding box of all 8s
    2. For each row with 8s, find its vertically symmetric row
    3. Extract values from the symmetric row at the same column positions
    4. Return only the filled 8s region
    """

    rows, cols = len(grid), len(grid[0])

    # Find bounding box of 8s
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    if min_row == rows:
        return [[]]

    result = []

    # For each row in the 8s region
    for row_idx in range(min_row, max_row + 1):
        row_result = []

        # Find vertically symmetric row
        sym_row = rows - 1 - row_idx

        # Extract values from the symmetric row at the same columns
        for col_idx in range(min_col, max_col + 1):
            value = grid[sym_row][col_idx]
            row_result.append(value)

        result.append(row_result)

    return result
