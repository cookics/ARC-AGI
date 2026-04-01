def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input has non-zero values arranged in anti-diagonal patterns.
    2. Each anti-diagonal (where i+j is constant) contains the same non-zero value.
    3. The output fills the entire grid with a repeating pattern based on (i+j) % k where k is the number of distinct non-zero values.

    Procedure:
    1. Find all non-zero values and group by anti-diagonal (i+j).
    2. Create mapping from (i+j) % k to values.
    3. Fill output grid using this pattern.
    """

    n = len(grid)
    m = len(grid[0])

    # Find non-zero values grouped by anti-diagonal
    antidiag_values = {}
    for i in range(n):
        for j in range(m):
            if grid[i][j] != 0:
                antidiag = i + j
                if antidiag not in antidiag_values:
                    antidiag_values[antidiag] = grid[i][j]
                else:
                    # Verify consistency
                    assert antidiag_values[antidiag] == grid[i][j]

    # Get number of unique values
    k = len(antidiag_values)

    # Create pattern mapping
    pattern = [0] * k
    for antidiag, value in antidiag_values.items():
        pattern[antidiag % k] = value

    # Fill output grid
    result = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            result[i][j] = pattern[(i + j) % k]

    return result
