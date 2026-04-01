def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an NxM grid containing various integer values.
    2. Output is an (N*N) x (N*M) grid where each input cell is expanded to an NxN block.
    3. For each unique value in the input, create an NxN pattern where the value appears at the same relative positions as in the original grid.
    4. All other positions in the pattern are filled with 0.
    5. Each cell in the input grid is replaced with the NxN pattern corresponding to its value.

    Procedure:
    1. Find all unique values and their positions in the input grid.
    2. Create NxN patterns for each unique value based on their positions.
    3. Expand the input grid by replacing each cell with its corresponding NxN pattern.
    """

    n = len(grid)
    m = len(grid[0])

    # Find positions of each unique value
    value_positions = {}
    for i in range(n):
        for j in range(m):
            val = grid[i][j]
            if val not in value_positions:
                value_positions[val] = []
            value_positions[val].append((i, j))

    # Create NxN patterns for each value
    patterns = {}
    for val, positions in value_positions.items():
        pattern = [[0 for _ in range(n)] for _ in range(n)]
        for r, c in positions:
            # Map position to NxN pattern
            pattern[r][c] = val
        patterns[val] = pattern

    # Create output grid
    output = [[0 for _ in range(n * m)] for _ in range(n * n)]

    # Fill output grid
    for i in range(n):
        for j in range(m):
            val = grid[i][j]
            pattern = patterns[val]
            for di in range(n):
                for dj in range(n):
                    output[n * i + di][n * j + dj] = pattern[di][dj]

    return output
