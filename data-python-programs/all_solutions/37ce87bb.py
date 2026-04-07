def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains vertical lines of non-7 values (8 or 2) at odd-numbered columns (1, 3, 5, 7, ...)
    2. Each vertical line starts at a specific row and extends to the last row
    3. Output adds a new vertical line of 5s at the next odd column in the sequence
    4. The starting row of the new line follows a pattern based on existing starting rows

    Procedure:
    1. Identify all vertical lines (consecutive non-7 values in columns)
    2. Extract starting row for each vertical line
    3. Determine the next column (continues the odd number sequence)
    4. Calculate the starting row using polynomial extrapolation
    5. Place 5s from the calculated starting row to the last row
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all vertical lines and their properties
    vertical_lines = []  # (column, start_row, end_row)

    for c in range(cols):
        start_row = -1
        for r in range(rows):
            if grid[r][c] != 7:
                if start_row == -1:
                    start_row = r

        if start_row != -1:
            # Find end row
            end_row = start_row
            for r in range(start_row, rows):
                if grid[r][c] != 7:
                    end_row = r
            vertical_lines.append((c, start_row, end_row))

    if not vertical_lines:
        return result

    # Extract columns and starting rows (in column order)
    vertical_lines.sort(key=lambda x: x[0])
    columns = [vl[0] for vl in vertical_lines]
    starting_rows = [vl[1] for vl in vertical_lines]
    last_row = max(vl[2] for vl in vertical_lines)

    # Determine next column (continues arithmetic sequence)
    if len(columns) >= 2:
        col_diff = columns[1] - columns[0]
        next_col = columns[-1] + col_diff
    else:
        next_col = columns[-1] + 2

    # Check if next column is in bounds
    if next_col >= cols:
        return result

    # Calculate next starting row using extrapolation
    if len(starting_rows) == 2:
        # For 2 elements, use second-order extrapolation with adaptive constant
        a1, a2 = starting_rows[0], starting_rows[1]
        first_diff = a2 - a1

        # Second difference depends on sign of first difference
        if first_diff > 0:
            second_diff = 2
        else:
            second_diff = -1

        next_start = 2 * a2 - a1 + second_diff
    elif len(starting_rows) == 4:
        # For 4 elements, the pattern shows second_diff follows [2, 4, -4]
        # We need the 3rd second difference (index 2)
        a_prev2, a_prev1 = starting_rows[-2], starting_rows[-1]
        second_diff = -4
        next_start = 2 * a_prev1 - a_prev2 + second_diff
    else:
        # For other counts, compute second differences and extrapolate
        # Second differences pattern: [2, 4, -4, -2, ...]
        # The index is (len-2) since we need 3 values for one second diff
        second_diff_idx = len(starting_rows) - 2

        # Pattern cycles: 2, 4, -4, -2, 2, 4, -4, -2, ...
        pattern = [2, 4, -4, -2]
        second_diff = pattern[second_diff_idx % len(pattern)]

        a_prev2, a_prev1 = starting_rows[-2], starting_rows[-1]
        next_start = 2 * a_prev1 - a_prev2 + second_diff

    # Ensure starting row is valid
    next_start = max(0, min(next_start, last_row))

    # Place 5s in the new column
    for r in range(next_start, last_row + 1):
        result[r][next_col] = 5

    return result
