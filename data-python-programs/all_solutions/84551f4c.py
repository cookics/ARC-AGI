def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3-row grid with values 0, 1, 2
    2. Output is a 3-row grid of the same dimensions
    3. Columns where all 3 rows have the same non-zero value undergo transformation
    4. Value 1 columns always "fall" to the bottom row and expand horizontally
    5. Value 2 columns fall if within distance ≤3 of any falling column (transitively)
    6. Falling columns create a 3-cell horizontal line in row 2 and clear rows 0,1
    7. Non-falling columns keep their value in all 3 rows

    Procedure:
    1. Identify columns where all 3 rows have the same non-zero value
    2. Separate into columns with value 1 and value 2
    3. Compute which columns fall using transitive closure (distance ≤3)
    4. Apply transformations to create output grid
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find columns with uniform non-zero values
    uniform_cols = {}  # col_idx -> value
    for c in range(cols):
        val = grid[0][c]
        if val != 0 and all(grid[r][c] == val for r in range(rows)):
            uniform_cols[c] = val

    # Determine which columns fall
    falling = set()

    # All value-1 columns fall
    for c, val in uniform_cols.items():
        if val == 1:
            falling.add(c)

    # Value-2 columns fall if within distance ≤3 of any falling column (transitive)
    changed = True
    while changed:
        changed = False
        for c, val in uniform_cols.items():
            if c not in falling and val == 2:
                for fc in falling:
                    if abs(c - fc) <= 3:
                        falling.add(c)
                        changed = True
                        break

    # Create output grid
    result = [[0] * cols for _ in range(rows)]

    # Apply transformations
    for c, val in uniform_cols.items():
        if c in falling:
            # Create 3-cell horizontal line in row 2 (bottom row)
            for i in range(3):
                if c + i < cols:
                    result[2][c + i] = val
        else:
            # Keep value in all rows
            for r in range(rows):
                result[r][c] = val

    return result
