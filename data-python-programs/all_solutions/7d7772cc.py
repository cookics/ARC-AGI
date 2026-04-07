def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    - The grid is divided by a line of identical values (column or row)
    - There are special values in specific positions on both sides of the divide
    - Compare corresponding positions and move values based on equality

    Procedure:
    1. Find the dividing line (column or row with all same values)
    2. Find the rows/columns with special values on both sides
    3. Compare corresponding positions and apply movement rules
    """

    result = [row[:] for row in grid]  # Deep copy
    rows, cols = len(grid), len(grid[0])

    # Find horizontal dividing line first (row with all same values)
    dividing_row = None
    for r in range(1, rows - 1):  # Skip first and last row
        if all(grid[r][c] == grid[r][0] for c in range(cols)):
            # Check if this row separates different sections
            above_val = grid[r - 1][0]
            below_val = grid[r + 1][0]
            row_val = grid[r][0]
            # Look for a row that's different from at least one neighbor
            if row_val != above_val:
                dividing_row = r
                break

    if dividing_row is not None:
        # Horizontal division found - handle as in training case 1
        if dividing_row > 0 and dividing_row < rows - 1:
            # Find rows with special values
            special_top_row = None
            special_bottom_row = None

            # Find top row with special values (first non-background row)
            top_bg = grid[0][0]
            for r in range(dividing_row):
                if any(grid[r][c] != top_bg for c in range(cols)):
                    special_top_row = r
                    break

            # Find bottom row with special values (first non-background row)
            bottom_bg = grid[dividing_row + 1][0]
            for r in range(dividing_row + 1, rows):
                if any(grid[r][c] != bottom_bg for c in range(cols)):
                    special_bottom_row = r
                    break

            if special_top_row is not None and special_bottom_row is not None:
                # Compare corresponding positions and apply rules
                for c in range(cols):
                    top_val = grid[special_top_row][c]
                    bottom_val = grid[special_bottom_row][c]

                    # Skip background positions
                    if top_val == top_bg:
                        continue

                    if top_val == bottom_val:
                        # Same values: move top value to row above dividing line
                        result[dividing_row - 1][c] = top_val
                    else:
                        # Different values: move top value to top of section
                        result[0][c] = top_val

                    # Clear the original position
                    result[special_top_row][c] = top_bg
        return result

    # Find vertical dividing line (column with all same values)
    dividing_col = None
    for c in range(cols):
        if all(grid[r][c] == grid[0][c] for r in range(rows)):
            dividing_col = c
            break

    if dividing_col is not None:
        # Vertical division found - handle as in training case 2
        if dividing_col > 0 and dividing_col < cols - 1:
            left_col = dividing_col - 1
            right_start = dividing_col + 1
            right_end = cols - 1

            # Find the column with special values in the right section
            special_right_col = None
            for c in range(right_start, cols):
                background_val = grid[0][right_start]
                if any(grid[r][c] != background_val for r in range(rows)):
                    special_right_col = c
                    break

            if special_right_col is not None:
                left_bg = grid[0][0]
                right_bg = grid[0][right_start]

                for r in range(rows):
                    left_val = grid[r][left_col]
                    right_val = grid[r][special_right_col]

                    if left_val == left_bg and right_val == right_bg:
                        continue

                    if left_val == right_val:
                        result[r][right_start] = left_val
                        result[r][special_right_col] = right_bg
                    else:
                        result[r][right_end] = right_val
                        result[r][special_right_col] = right_bg

    return result
