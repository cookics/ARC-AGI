def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing integers (0s and non-zero values).
    2. Output is the same size grid where non-zero values have "fallen down" due to gravity.
    3. Each column operates independently - non-zero values collect at the bottom.
    4. Non-zero values preserve their relative order from top to bottom.

    Procedure:
    1. For each column, collect all non-zero values from top to bottom.
    2. Place them at the bottom of the column, filling from bottom up.
    3. Fill remaining positions at top with zeros.
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    for col in range(cols):
        # Collect non-zero values from this column (top to bottom)
        non_zero_values = []
        for row in range(rows):
            if grid[row][col] != 0:
                non_zero_values.append(grid[row][col])

        # Place them at the bottom of the result column
        for i, value in enumerate(non_zero_values):
            result[rows - len(non_zero_values) + i][col] = value

    return result
