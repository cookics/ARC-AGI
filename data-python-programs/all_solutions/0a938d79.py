def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing exactly two non-zero values at different positions
    2. Output is a grid of the same size with a repeating pattern based on the input values
    3. Pattern type depends on the relative distances between the two non-zero positions
    4. If column distance is less than row distance, creates vertical stripes repeating horizontally
    5. Otherwise, fills entire rows with values repeating vertically
    6. Repeat interval equals 2 times the relevant distance (column or row distance)

    Procedure:
    1. Find the two non-zero values and their positions in the input grid
    2. Calculate the row distance and column distance between the two positions
    3. Determine pattern type by comparing column distance to row distance
    4. Create output grid with appropriate repeating pattern based on determined type
    5. Fill the output grid using the calculated repeat interval and original positions
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find non-zero values and their positions
    non_zeros = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                non_zeros.append((i, j, grid[i][j]))

    if len(non_zeros) != 2:
        return grid  # Should have exactly 2 non-zero values

    (r1, c1, val1), (r2, c2, val2) = non_zeros

    row_distance = abs(r2 - r1)
    col_distance = abs(c2 - c1)

    # Create output grid
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    if col_distance > 0 and col_distance < row_distance:
        # Column-based pattern (vertical stripes)
        repeat_interval = 2 * col_distance

        for i in range(rows):
            for j in range(cols):
                # Check if this column matches one of the original columns in the repeating pattern
                if repeat_interval > 0:
                    if (j - c1) % repeat_interval == 0 and j >= c1:
                        result[i][j] = val1
                    elif (j - c2) % repeat_interval == 0 and j >= c2:
                        result[i][j] = val2
                else:
                    # Special case when repeat_interval is 0 (same column)
                    if j == c1:
                        result[i][j] = val1
                    elif j == c2:
                        result[i][j] = val2
    else:
        # Row-based pattern (fill entire rows)
        repeat_interval = 2 * row_distance

        for i in range(rows):
            # Check if this row matches one of the original rows in the repeating pattern
            if repeat_interval > 0:
                if (i - r1) % repeat_interval == 0 and i >= r1:
                    for j in range(cols):
                        result[i][j] = val1
                elif (i - r2) % repeat_interval == 0 and i >= r2:
                    for j in range(cols):
                        result[i][j] = val2
            else:
                # Special case when repeat_interval is 0 (same row)
                if i == r1:
                    for j in range(cols):
                        result[i][j] = val1
                elif i == r2:
                    for j in range(cols):
                        result[i][j] = val2

    return result
