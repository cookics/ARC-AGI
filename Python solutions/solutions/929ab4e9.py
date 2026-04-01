def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 24x24 grid with some cells containing value 2 (corrupted data)
    2. Output is the same grid with all 2s replaced by the correct values
    3. The grid has both horizontal and vertical reflection symmetry
    4. For position (i,j), symmetric positions are: (n-1-i, j), (i, n-1-j), (n-1-i, n-1-j)

    Procedure:
    1. Create a copy of the input grid
    2. Iterate through each cell in the grid
    3. If a cell contains value 2, check its three symmetric positions
    4. Replace the 2 with the first non-2 value found from symmetric positions
    5. Return the restored grid
    """

    # Create a copy of the grid
    result = [row[:] for row in grid]
    n = len(grid)

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 2:
                # Try to find a non-2 value from symmetric positions
                symmetric_positions = [
                    (n - 1 - i, j),  # horizontal symmetry
                    (i, n - 1 - j),  # vertical symmetry
                    (n - 1 - i, n - 1 - j),  # both symmetries
                ]

                for si, sj in symmetric_positions:
                    if grid[si][sj] != 2:
                        result[i][j] = grid[si][sj]
                        break

    return result
