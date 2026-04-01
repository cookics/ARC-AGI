def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid where the first column contains non-zero values and all other cells are zeros.
    2. The output preserves the first column unchanged.
    3. A diagonal line of 2s is created starting from the top-right corner, moving one column left each row down.
    4. The last row gets filled with 4s except for the leftmost column.

    Procedure:
    1. Copy the input grid to preserve the original structure.
    2. For each row except the last, place a 2 at position (width - 1 - row_index) if the column is not the first.
    3. For the last row, fill all positions except the first column with 4s.
    4. Return the modified grid.
    """

    height = len(grid)
    width = len(grid[0])
    result = [row[:] for row in grid]  # Copy the grid

    # Create diagonal of 2s for all rows except the last
    for row in range(height - 1):
        col = width - 1 - row
        if col >= 1:  # Don't overwrite the first column
            result[row][col] = 2

    # Fill the last row with 4s (except first column)
    for col in range(1, width):
        result[height - 1][col] = 4

    return result
