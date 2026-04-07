def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is an 8x8 grid containing non-zero elements that form a pattern.
    2. The output is a 3x6 grid.
    3. The pattern in the input fits within a 3x3 bounding box.
    4. The output is created by duplicating this 3x3 pattern horizontally.

    Procedure:
    1. Find all non-zero elements in the input grid.
    2. Extract the 3x3 bounding box that contains all non-zero elements.
    3. Duplicate this 3x3 pattern horizontally to create the 3x6 output.
    """

    # Find all non-zero positions
    non_zero_positions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] != 0:
                non_zero_positions.append((i, j))

    # Find bounding box
    min_row = min(pos[0] for pos in non_zero_positions)
    max_row = max(pos[0] for pos in non_zero_positions)
    min_col = min(pos[1] for pos in non_zero_positions)
    max_col = max(pos[1] for pos in non_zero_positions)

    # Extract the 3x3 pattern
    pattern = []
    for i in range(min_row, min_row + 3):
        row = []
        for j in range(min_col, min_col + 3):
            row.append(grid[i][j])
        pattern.append(row)

    # Duplicate horizontally to create 3x6 output
    result = []
    for i in range(3):
        result_row = pattern[i] + pattern[i]
        result.append(result_row)

    return result
