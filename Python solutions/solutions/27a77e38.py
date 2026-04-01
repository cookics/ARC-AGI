def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a separator row of all 5s at the middle row
    2. Output is the same grid with one modification in the last row
    3. The value placed comes from a diagonal position [middle_row-1, middle_col-1]
    4. This value is placed at position [last_row, middle_col]

    Procedure:
    1. Find the middle row and middle column indices
    2. Extract the diagonal element at [middle_row-1, middle_col-1]
    3. Place this value at the middle column of the last row
    """

    # Create a copy of the input grid
    result = [row[:] for row in grid]

    # Find the middle position
    rows = len(grid)
    cols = len(grid[0])
    middle_row = rows // 2
    middle_col = cols // 2

    # Verify that the middle row is filled with 5s
    assert all(cell == 5 for cell in grid[middle_row]), "Middle row should be all 5s"

    # Get the diagonal element just before the middle
    diagonal_element = grid[middle_row - 1][middle_col - 1]

    # Place this value in the middle column of the last row
    result[rows - 1][middle_col] = diagonal_element

    return result
