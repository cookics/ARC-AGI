def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3x3 grid with integer values (0, 1, or 2)
    2. Output is also a 3x3 grid with the same values
    3. Each row in the input shifts down by one position in the output
    4. A new row of all zeros appears at the top (row 0)
    5. The bottom row from the input disappears
    6. This creates a "gravity" effect where everything falls down one row

    Procedure:
    1. Create a new grid with the same dimensions
    2. Fill the first row with zeros
    3. Copy each input row to the next row down in the output
    4. The last input row gets discarded
    """
    rows = len(grid)
    cols = len(grid[0])

    # Create result grid
    result = [[0] * cols for _ in range(rows)]

    # First row is all zeros
    result[0] = [0] * cols

    # Shift all other rows down by one position
    for i in range(rows - 1):
        result[i + 1] = grid[i][:]

    return result
