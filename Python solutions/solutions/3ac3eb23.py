def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid with non-zero values only in the first row, rest are zeros.
    2. The output extends these values vertically through all rows following an alternating pattern.
    3. For even-indexed rows (0, 2, 4...), non-zero values remain in their original columns.
    4. For odd-indexed rows (1, 3, 5...), non-zero values are placed in adjacent columns (left and right).
    5. Values from column c appear at columns c-1 and c+1 in odd rows, respecting grid boundaries.

    Procedure:
    1. Initialize a result grid filled with zeros matching the input dimensions.
    2. Extract all non-zero values and their column positions from the first row.
    3. Iterate through each row and apply the alternating placement pattern.
    4. For even rows, place each value at its original column position.
    5. For odd rows, place each value at the adjacent left and right columns if within bounds.
    6. Return the transformed grid with the vertical repetition pattern applied.
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find non-zero values in the first row
    first_row_values = []
    for c in range(cols):
        if grid[0][c] != 0:
            first_row_values.append((c, grid[0][c]))

    # Apply the pattern to all rows
    for r in range(rows):
        for col, value in first_row_values:
            if r % 2 == 0:  # Even row
                result[r][col] = value
            else:  # Odd row
                # Place at col-1 and col+1 if within bounds
                if col - 1 >= 0:
                    result[r][col - 1] = value
                if col + 1 < cols:
                    result[r][col + 1] = value

    return result
