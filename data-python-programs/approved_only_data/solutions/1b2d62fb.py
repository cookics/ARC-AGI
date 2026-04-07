def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input grid has a column of 1s at index 3 that acts as a separator.
    2. Left side is columns 0-2, right side is columns 4-6.
    3. The output has 8 at position (i,j) if both left[i][j]==0 and right[i][j]==0, otherwise 0.

    Procedure:
    1. Split the input grid into left (columns 0-2) and right (columns 4-6) sides.
    2. For each position, check if both sides have 0 at that position.
    3. If yes, output 8; otherwise output 0.
    """

    rows = len(grid)
    result = []

    for i in range(rows):
        row = []
        for j in range(3):  # output has 3 columns
            left_val = grid[i][j]  # columns 0-2
            right_val = grid[i][j + 4]  # columns 4-6

            if left_val == 0 and right_val == 0:
                row.append(8)
            else:
                row.append(0)
        result.append(row)

    return result
