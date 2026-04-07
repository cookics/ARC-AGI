def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 10x10 grid with exactly 2 non-zero values at different positions.
    2. The output creates rectangular border patterns using these non-zero values.
    3. The grid is conceptually split into upper half (rows 0-4) and lower half (rows 5-9).
    4. Upper half value fills row 0, its own row, and columns 0,9 for rows 0-4.
    5. Lower half value fills row 9, its own row, and columns 0,9 for rows 5-9.

    Procedure:
    1. Find the two non-zero values and their positions.
    2. Create output grid initialized with zeros.
    3. For upper half value: fill border pattern in rows 0-4.
    4. For lower half value: fill border pattern in rows 5-9.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find the two non-zero values
    non_zero_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero_positions.append((r, c, grid[r][c]))

    # Process each non-zero value
    for row, col, value in non_zero_positions:
        if row <= 4:  # Upper half (rows 0-4)
            # Fill row 0 completely
            for c in range(cols):
                result[0][c] = value
            # Fill the row where value is located
            for c in range(cols):
                result[row][c] = value
            # Fill columns 0 and 9 for rows 0-4
            for r in range(5):
                result[r][0] = value
                result[r][cols - 1] = value
        else:  # Lower half (rows 5-9)
            # Fill row 9 completely
            for c in range(cols):
                result[rows - 1][c] = value
            # Fill the row where value is located
            for c in range(cols):
                result[row][c] = value
            # Fill columns 0 and 9 for rows 5-9
            for r in range(5, rows):
                result[r][0] = value
                result[r][cols - 1] = value

    return result
