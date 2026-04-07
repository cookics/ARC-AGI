def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is 3x4, output is 6x8 (doubled dimensions).
    2. Each output row is created by reversing the input row and concatenating with the original row.
    3. Rows are arranged in a vertically mirrored pattern: [last, middle, first, first, middle, last].
    4. For 3 input rows, the output order is [row2, row1, row0, row0, row1, row2].
    5. This creates a symmetric transformation where the grid is reflected both horizontally and vertically.

    Procedure:
    1. For each input row, create expanded_row = reversed_row + original_row.
    2. Stack expanded rows in pattern: [last, middle, first, first, middle, last].
    """

    rows = len(grid)

    # Create expanded rows
    expanded_rows = []
    for row in grid:
        reversed_row = row[::-1]  # Reverse the row
        expanded_row = reversed_row + row  # Concatenate reversed + original
        expanded_rows.append(expanded_row)

    # Stack in the pattern: [last, middle, first, first, middle, last]
    # For 3 rows: [row2, row1, row0, row0, row1, row2]
    result = []

    # Add rows in reverse order (bottom to top)
    for i in range(rows - 1, -1, -1):
        result.append(expanded_rows[i])

    # Add rows in original order (top to bottom)
    for i in range(rows):
        result.append(expanded_rows[i])

    return result
