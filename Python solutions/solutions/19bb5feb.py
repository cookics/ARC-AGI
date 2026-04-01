def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input grid contains a rectangular region filled with 8s, surrounded by 0s.
    2. Inside the 8s region, there are 2x2 colored blocks (values other than 0 and 8).
    3. The 8s region is divided into 4 quadrants, and each quadrant's color (if any) is mapped to the corresponding position in a 2x2 output grid.
    4. If no color exists in a quadrant, 0 is placed in the output.

    Procedure:
    1. Find the bounds of the 8s region
    2. Divide the region into 4 quadrants
    3. For each quadrant, find any colored block (non-0, non-8)
    4. Create a 2x2 output with the colors from each quadrant
    """

    # Find the bounds of the 8s region
    min_row, max_row = float("inf"), -1
    min_col, max_col = float("inf"), -1

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 8:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    # Calculate the midpoints to divide into quadrants
    mid_row = (min_row + max_row + 1) // 2
    mid_col = (min_col + max_col + 1) // 2

    # Function to find color in a quadrant
    def find_color_in_quadrant(row_start, row_end, col_start, col_end):
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                if i < len(grid) and j < len(grid[0]):
                    if grid[i][j] != 0 and grid[i][j] != 8:
                        return grid[i][j]
        return 0

    # Find colors in each quadrant
    top_left = find_color_in_quadrant(min_row, mid_row, min_col, mid_col)
    top_right = find_color_in_quadrant(min_row, mid_row, mid_col, max_col + 1)
    bottom_left = find_color_in_quadrant(mid_row, max_row + 1, min_col, mid_col)
    bottom_right = find_color_in_quadrant(mid_row, max_row + 1, mid_col, max_col + 1)

    # Create the 2x2 output
    return [[top_left, top_right], [bottom_left, bottom_right]]
