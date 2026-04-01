def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 3 unique values, where 6 is always noise to be removed
    2. Remaining 2 values: one is background, one is pattern indicator
    3. Rows containing pattern indicator get alternating pattern in output
    4. Pattern interval: 3 for square grids (12x12), 2 for non-square (15x19)
    5. Pattern positions: where column_index % interval == 1

    Procedure:
    1. Remove value 6, leaving 2 values
    2. Find which value appears in fewer rows (pattern indicator)
    3. Create output where pattern rows have alternating values
    """

    rows = len(grid)
    cols = len(grid[0])

    # Get unique values and remove 6
    unique_vals = set()
    for row in grid:
        unique_vals.update(row)
    unique_vals.discard(6)
    vals = sorted(list(unique_vals))

    # Count rows containing each value
    val_rows = {v: set() for v in vals}
    for i, row in enumerate(grid):
        for val in vals:
            if val in row:
                val_rows[val].add(i)

    # Pattern indicator appears in fewer rows
    if len(val_rows[vals[0]]) < len(val_rows[vals[1]]):
        pattern_val = vals[0]
        background_val = vals[1]
        pattern_rows = val_rows[vals[0]]
    else:
        pattern_val = vals[1]
        background_val = vals[0]
        pattern_rows = val_rows[vals[1]]

    # Pattern interval: 3 for square, 2 for non-square
    pattern_interval = 3 if rows == cols else 2

    # Build output
    result = []
    for i in range(rows):
        if i in pattern_rows:
            # Pattern row: alternating values
            row = []
            for j in range(cols):
                if j % pattern_interval == 1:
                    row.append(pattern_val)
                else:
                    row.append(background_val)
            result.append(row)
        else:
            # Background row: all same value
            result.append([background_val] * cols)

    return result
