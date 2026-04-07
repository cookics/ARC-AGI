def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a repeating diagonal pattern with period N (2 or 3 values cycling diagonally)
    2. Separator value forms borders on right and/or bottom edges
    3. Output extends the pattern to entire grid, shifted by 1 position
    4. Pattern formula: value at (r,c) determined by (r+c) % period

    Procedure:
    1. Identify separator value from bottom-right corner
    2. Extract non-separator values to determine period
    3. Build pattern array mapping (r+c)%period to values
    4. Generate output with pattern shifted by 1: pattern[(r+c+1)%period]
    """

    rows = len(grid)
    cols = len(grid[0])

    # Identify separator value (bottom-right corner)
    separator = grid[rows-1][cols-1]

    # Extract all non-separator values
    values_set = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != separator:
                values_set.add(grid[r][c])

    period = len(values_set)

    # Build the repeating pattern
    # pattern[k] = value at positions where (r+c) % period == k
    pattern = [None] * period
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != separator:
                index = (r + c) % period
                pattern[index] = grid[r][c]

    # Create output with pattern shifted by 1
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            result[r][c] = pattern[(r + c + 1) % period]

    return result
