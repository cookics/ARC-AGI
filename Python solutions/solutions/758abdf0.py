def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with either an entire row or column filled with 0s (acting as a reference line)
    2. Output is the same grid with 8s redistributed based on the reference line orientation
    3. If first column is all 0s: process rows horizontally
    4. If first/last row is all 0s: process columns vertically
    5. Single 8 in a line → duplicate it (next position for rows, next row for columns)
    6. Multiple 8s in a line → remove them and place same number of 0s at the far end

    Procedure:
    1. Copy the input grid to result
    2. Detect which orientation (row or column) has the 0s reference line
    3. For horizontal processing (first column all 0s): iterate through each row
    4. For vertical processing (first/last row all 0s): iterate through each column
    5. Count the 8s in each line (row or column)
    6. If count is 1: duplicate the 8 to adjacent position
    7. If count > 1: replace all 8s with 7s and place 0s at the far end
    8. Return the modified grid
    """

    result = [row[:] for row in grid]

    # Check if first row is all 0s (column-based transformation)
    if all(val == 0 for val in grid[0]):
        # Column-based transformation
        for col in range(len(grid[0])):
            # Find 8s in this column
            eight_rows = [row for row in range(len(grid)) if grid[row][col] == 8]

            if len(eight_rows) == 1:
                # Single 8 in column: copy it to the next row
                row = eight_rows[0]
                if row + 1 < len(grid):
                    result[row + 1][col] = 8
            elif len(eight_rows) > 1:
                # Multiple 8s in column: remove all and place 0s at bottom
                for row in eight_rows:
                    result[row][col] = 7
                # Place 0s at bottom
                for i in range(len(eight_rows)):
                    bottom_row = len(grid) - 1 - i
                    if bottom_row >= 0:
                        result[bottom_row][col] = 0

    # Check if last row is all 0s (also column-based transformation)
    elif all(val == 0 for val in grid[-1]):
        # Column-based transformation with bottom reference
        for col in range(len(grid[0])):
            # Find 8s in this column
            eight_rows = [row for row in range(len(grid)) if grid[row][col] == 8]

            if len(eight_rows) == 1:
                # Single 8 in column: copy it downward
                row = eight_rows[0]
                if row + 1 < len(grid):
                    result[row + 1][col] = 8
            elif len(eight_rows) > 1:
                # Multiple 8s in column: remove all and place 0s where the 0s were
                for row in eight_rows:
                    result[row][col] = 7
                # Keep the 0s in the bottom row (they're already there)

    # Check if first column is all 0s (row-based transformation)
    elif all(grid[row][0] == 0 for row in range(len(grid))):
        # Row-based transformation
        for row_idx, row in enumerate(grid):
            eight_positions = [i for i, val in enumerate(row) if val == 8]

            if len(eight_positions) == 1:
                # Single 8: duplicate it to the next position
                pos = eight_positions[0]
                if pos + 1 < len(row):
                    result[row_idx][pos + 1] = 8
            elif len(eight_positions) > 1:
                # Multiple 8s: replace with 7s and add 0s at end
                for pos in eight_positions:
                    result[row_idx][pos] = 7
                # Place 0s at the end
                for i in range(len(eight_positions)):
                    if len(row) - 1 - i >= 0:
                        result[row_idx][len(row) - 1 - i] = 0

    return result
