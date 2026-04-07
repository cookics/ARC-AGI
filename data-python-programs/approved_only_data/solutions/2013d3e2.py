def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 10x10 grid containing mostly zeros with some non-zero colored values forming patterns.
    2. The non-zero values create connected shapes or structures within the grid.
    3. The output is always a 3x3 grid extracted from a specific region of the input.
    4. The output represents the top-left 3x3 portion of the bounding box that contains all non-zero values.
    5. In the first example, the bounding box starts at (1,1) yielding output [[1,0,0],[0,3,6],[0,5,2]].
    6. In the second example, the bounding box starts at (2,2) yielding output [[0,0,7],[0,6,8],[7,8,4]].

    Procedure:
    1. Scan through the entire 10x10 grid to locate all positions containing non-zero values.
    2. Calculate the minimum and maximum row and column indices to determine the bounding rectangle.
    3. Extract a 3x3 subgrid starting from the top-left corner (min_row, min_col) of this bounding box.
    4. Handle boundary conditions by filling with zeros if the 3x3 region extends beyond grid limits.
    5. Return the resulting 3x3 grid as the final output.
    """

    # Find bounding box of non-zero values
    min_row, max_row = len(grid), -1
    min_col, max_col = len(grid[0]), -1

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    # Extract 3x3 region from top-left of bounding box
    result = []
    for r in range(3):
        row = []
        for c in range(3):
            row_idx = min_row + r
            col_idx = min_col + c
            if row_idx < len(grid) and col_idx < len(grid[0]):
                row.append(grid[row_idx][col_idx])
            else:
                row.append(0)
        result.append(row)

    return result
