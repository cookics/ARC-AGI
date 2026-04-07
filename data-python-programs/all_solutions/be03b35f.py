def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 5×5 grid partitioned by row 2 and column 2 (all zeros) into four 2×2 quadrants
    2. Bottom-right quadrant (rows 3-4, cols 3-4) is always [[2,2],[2,2]] (marker)
    3. Output is a 2×2 grid computed from the other three quadrants
    4. The computation uses specific position-dependent formulas combining top-left and bottom-left quadrants

    Procedure:
    1. Extract top-left quadrant (TL): rows 0-1, cols 0-1
    2. Extract bottom-left quadrant (BL): rows 3-4, cols 0-1
    3. Compute output using position-specific formulas:
       - Out[0][0] = NOT BL[0][0]
       - Out[0][1] = NOT TL[0][1]
       - Out[1][0] = TL[1][0] OR BL[1][0]
       - Out[1][1] = BL[0][1]
    """

    # Extract top-left quadrant (rows 0-1, cols 0-1)
    TL = [[grid[0][0], grid[0][1]],
          [grid[1][0], grid[1][1]]]

    # Extract bottom-left quadrant (rows 3-4, cols 0-1)
    BL = [[grid[3][0], grid[3][1]],
          [grid[4][0], grid[4][1]]]

    # Compute output using position-specific formulas
    result = [[0, 0], [0, 0]]

    result[0][0] = 1 - BL[0][0]  # NOT BL[0][0]
    result[0][1] = 1 - TL[0][1]  # NOT TL[0][1]
    result[1][0] = TL[1][0] | BL[1][0]  # TL[1][0] OR BL[1][0]
    result[1][1] = BL[0][1]  # BL[0][1]

    return result
