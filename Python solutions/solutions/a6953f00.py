def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a square or rectangular grid of varying dimensions.
    2. Output is always a 2×2 subgrid extracted from the first 2 rows.
    3. Column selection depends on grid width parity.
    4. If width is even: extract columns starting at width//2 (right half).
    5. If width is odd: extract columns starting at 0 (left side).

    Procedure:
    1. Determine the width of the input grid.
    2. Calculate starting column: width//2 if width is even, else 0.
    3. Extract 2×2 subgrid from rows [0:2] and columns [start_col:start_col+2].
    """

    width = len(grid[0])

    if width % 2 == 0:
        # Even width: take right half columns
        start_col = width // 2
    else:
        # Odd width: take left columns
        start_col = 0

    # Extract 2x2 subgrid from rows 0,1 and cols start_col, start_col+1
    result = []
    for row in range(2):
        result_row = []
        for col in range(start_col, start_col + 2):
            result_row.append(grid[row][col])
        result.append(result_row)

    return result
