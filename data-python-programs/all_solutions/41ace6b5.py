def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a row where even columns contain 2 (row_2)
    2. The next row (row_5 = row_2 + 1) has 5 at even columns
    3. Even columns remain mostly unchanged (structural columns)
    4. Odd columns have a fixed structure based on row_2:
       - A section of 8s ending at row_2
       - A section of 1s starting at row_5
       - 7s fill the top, 9s fill the bottom

    Procedure:
    1. Find row_2 (row where column 0 has value 2)
    2. Calculate the size of the 8-section and 1-section based on row_2
    3. For each odd column:
       - Count 1s in the input
       - Fill the fixed 8-section with 8s
       - Fill 1-section based on count (with a cap)
       - Fill remaining with 7s (top) and 9s (bottom)
    """

    # Find row_2 (the row where even columns have value 2)
    row_2 = None
    for r in range(len(grid)):
        if grid[r][0] == 2:
            row_2 = r
            break

    row_5 = row_2 + 1
    num_rows = len(grid)
    num_cols = len(grid[0])

    # Calculate the fixed structure sizes based on row_2
    num_8_rows = row_2 + 1 if row_2 <= 2 else row_2 // 2
    max_1_rows = float('inf') if row_2 <= 4 else num_8_rows - 1

    # Initialize output grid as copy of input
    output = [row[:] for row in grid]

    # Process each odd column (1, 3, 5, ...)
    for c in range(1, num_cols, 2):
        # Count 1s in this column
        count_1 = sum(1 for r in range(num_rows) if grid[r][c] == 1)

        # If row_5 has 8, the cap doesn't apply (the 8 becomes a 1)
        if grid[row_5][c] == 8:
            num_1_rows = count_1
        else:
            num_1_rows = min(count_1, max_1_rows)

        # Calculate positions
        start_8 = row_2 - num_8_rows + 1
        end_1 = row_5 + num_1_rows - 1

        # Fill with 7 at the top (before 8s start)
        for r in range(start_8):
            output[r][c] = 7

        # Fill with 8 (fixed section)
        for r in range(start_8, row_2 + 1):
            output[r][c] = 8

        # Fill with 1
        for r in range(row_5, end_1 + 1):
            output[r][c] = 1

        # Fill with 9 at the bottom
        for r in range(end_1 + 1, num_rows):
            output[r][c] = 9

    return output
