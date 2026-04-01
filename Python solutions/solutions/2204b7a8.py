def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing integers including the value 3.
    2. The output is the same grid but with all 3s replaced by border colors.
    3. The grid has either horizontal borders (uniform top and bottom rows) or vertical borders (uniform left and right columns).
    4. When horizontal borders exist, 3s are replaced with the color of the closer border (top or bottom row).
    5. When vertical borders exist, 3s are replaced with the color of the closer border (left or right column).

    Procedure:
    1. Create a deep copy of the input grid to store results.
    2. Check if horizontal borders exist by verifying top and bottom rows are uniform.
    3. Check if vertical borders exist by verifying left and right columns are uniform.
    4. For each cell containing 3, calculate distance to relevant borders.
    5. Replace each 3 with the color of the closer border based on the border type.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Check for horizontal borders (top/bottom rows uniform)
    top_uniform = len(set(grid[0])) == 1
    bottom_uniform = len(set(grid[rows - 1])) == 1
    horizontal_borders = top_uniform and bottom_uniform

    # Check for vertical borders (left/right columns uniform)
    left_col = [grid[i][0] for i in range(rows)]
    right_col = [grid[i][cols - 1] for i in range(rows)]
    left_uniform = len(set(left_col)) == 1
    right_uniform = len(set(right_col)) == 1
    vertical_borders = left_uniform and right_uniform

    # Process each cell
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 3:
                if horizontal_borders:
                    # Distance to top vs bottom
                    dist_top = i
                    dist_bottom = rows - 1 - i
                    if dist_top <= dist_bottom:
                        result[i][j] = grid[0][0]  # Top border color
                    else:
                        result[i][j] = grid[rows - 1][0]  # Bottom border color

                elif vertical_borders:
                    # Distance to left vs right
                    dist_left = j
                    dist_right = cols - 1 - j
                    if dist_left <= dist_right:
                        result[i][j] = grid[0][0]  # Left border color
                    else:
                        result[i][j] = grid[0][cols - 1]  # Right border color

    return result
