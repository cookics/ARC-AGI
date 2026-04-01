def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid where some rows have value 5 in the leftmost column
    2. Output preserves the original content (non-empty rows)
    3. After the first completely empty row, the pattern repeats cyclically
    4. The repeating pattern is the rows that start with 5, but with 5 replaced by 0

    Procedure:
    1. Find all rows that start with 5 in the leftmost column
    2. Create modified versions of these rows with the leftmost 5 replaced by 0
    3. Find the first completely empty row in the grid
    4. Fill remaining rows by cycling through the modified pattern rows
    """

    # Create a copy of the original grid
    result = [row[:] for row in grid]

    # Find rows that start with 5
    rows_with_5 = []
    for i, row in enumerate(grid):
        if row[0] == 5:
            # Create a copy of the row with leftmost 5 replaced by 0
            modified_row = row[:]
            modified_row[0] = 0
            rows_with_5.append(modified_row)

    # Find the first completely empty row
    first_empty_row = len(grid)
    for i, row in enumerate(grid):
        if all(cell == 0 for cell in row):
            first_empty_row = i
            break

    # Fill empty rows with the modified rows (cycling through them)
    if rows_with_5:
        row_index = 0
        for i in range(first_empty_row, len(grid)):
            result[i] = rows_with_5[row_index % len(rows_with_5)][:]
            row_index += 1

    return result
