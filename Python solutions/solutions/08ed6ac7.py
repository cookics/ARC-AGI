def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing 0s and 5s where 5s form vertical columns.
    2. Output is a 2D grid with the same dimensions where vertical columns of 5s are replaced with numbers.
    3. The longest vertical column of consecutive 5s gets replaced with 1.
    4. The second longest column gets replaced with 2, and so on.
    5. All other cells remain 0.

    Procedure:
    1. Identify all vertical columns of consecutive 5s in the grid.
    2. Calculate the length of each vertical column.
    3. Sort the columns by their length in descending order.
    4. Assign rank numbers (1, 2, 3, ...) to each column based on their sorted order.
    5. Replace each column of 5s with its corresponding rank number in the result grid.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all vertical columns of 5s
    columns = []  # List of (col_index, start_row, end_row, length)

    for c in range(cols):
        start_row = None

        for r in range(rows):
            if grid[r][c] == 5:
                if start_row is None:
                    start_row = r
            else:
                if start_row is not None:
                    # End of a column of 5s
                    end_row = r - 1
                    length = end_row - start_row + 1
                    columns.append((c, start_row, end_row, length))
                    start_row = None

        # Check if column ends with 5s
        if start_row is not None:
            end_row = rows - 1
            length = end_row - start_row + 1
            columns.append((c, start_row, end_row, length))

    # Sort columns by length (descending) to assign ranks
    columns.sort(key=lambda x: x[3], reverse=True)

    # Assign values 1, 2, 3, 4... based on rank
    for rank, (col_index, start_row, end_row, length) in enumerate(columns):
        value = rank + 1

        # Fill the column with the assigned value
        for r in range(start_row, end_row + 1):
            result[r][col_index] = value

    return result
