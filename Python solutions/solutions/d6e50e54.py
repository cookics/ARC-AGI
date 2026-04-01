def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Find a rectangular region of 1s in the grid
    2. Change all 1s to 2s
    3. Find all 9s in the grid
    4. For each 9, if it shares a row or column with the rectangle, project it to that intersection point within the rectangle

    Procedure:
    1. Find the bounding box of all 1s (rectangle coordinates)
    2. Create output grid by copying input and changing 1s to 2s
    3. Find all 9 positions
    4. For each 9, if its row intersects rectangle, place at (same_row, intersection_with_rectangle)
    5. If its column intersects rectangle, place at (intersection_with_rectangle, same_col)
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy input grid

    # Find rectangle of 1s
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    # Change all 1s to 2s
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                result[i][j] = 2

    # Find all 9s and project them into the rectangle
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 9:
                # Check if this 9's row or column intersects with rectangle
                if min_row <= i <= max_row and min_col <= j <= max_col:
                    # 9 is already inside rectangle, keep it
                    continue
                elif min_row <= i <= max_row:
                    # Same row as rectangle, project beyond rectangle edge
                    if j < min_col:
                        # 9 is to the left of rectangle, project one cell beyond left edge
                        if min_col - 1 >= 0:
                            result[i][min_col - 1] = 9
                    else:
                        # 9 is to the right of rectangle, project one cell beyond right edge
                        if max_col + 1 < cols:
                            result[i][max_col + 1] = 9
                elif min_col <= j <= max_col:
                    # Same column as rectangle, project beyond rectangle edge
                    if i < min_row:
                        # 9 is above rectangle, project to top edge
                        if min_row < rows:
                            result[min_row][j] = 9
                    else:
                        # 9 is below rectangle, project based on column position relative to rectangle center
                        rect_center_col = (min_col + max_col) / 2
                        if j <= rect_center_col:
                            # Left side columns: project to bottom edge of rectangle
                            result[max_row][j] = 9
                        else:
                            # Right side columns: project one row beyond bottom edge
                            if max_row + 1 < rows:
                                result[max_row + 1][j] = 9

                # Remove the original 9 (if it was moved)
                if not (min_row <= i <= max_row and min_col <= j <= max_col):
                    result[i][j] = 7

    return result
