def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid filled entirely with 0s of varying dimensions.
    2. The output has the same dimensions as the input grid.
    3. The pattern transforms the grid by setting all perimeter/border cells to 8.
    4. Interior cells (not on the border) remain as 0.
    5. This creates a rectangular frame or border pattern with 8s forming the outline.

    Procedure:
    1. Create a copy of the input grid to work with.
    2. Iterate through all cells in the grid.
    3. For cells on the perimeter (first/last row or first/last column), set value to 8.
    4. Keep all interior cells unchanged (as 0).
    5. Return the transformed grid.
    """

    rows = len(grid)
    cols = len(grid[0])

    # Create result grid, initially copy of input
    result = [row[:] for row in grid]

    # Set perimeter cells to 8
    for i in range(rows):
        for j in range(cols):
            # If on perimeter (first/last row or first/last column)
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                result[i][j] = 8

    return result
