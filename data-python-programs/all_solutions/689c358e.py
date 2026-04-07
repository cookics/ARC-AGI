def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an 11×11 grid with colored shapes (values 2, 5, 8, 9) on a background of 6s and 7s
    2. Output is the same grid with projections of each colored shape onto the borders
    3. Projection direction depends on the centroid position of each colored shape
    4. If centroid row < 5: project vertically to top/bottom borders
    5. If centroid row >= 5: project horizontally to left/right borders based on centroid column

    Procedure:
    1. Copy the input grid to result
    2. For each color (2, 5, 8, 9), find all positions of that color
    3. Calculate the centroid (average row, average column) of positions
    4. If centroid row < 5:
       - Place color at (0, round(centroid_col))
       - Place 0 at (10, round(centroid_col))
    5. If centroid row >= 5:
       - If centroid col < 5.5: place color at (round(centroid_row), 0) and 0 at (round(centroid_row), 10)
       - If centroid col >= 5.5: place 0 at (round(centroid_row), 0) and color at (round(centroid_row), 10)
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Process colors 2, 5, 8, 9
    for color in [2, 5, 8, 9]:
        # Find all positions of this color
        positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    positions.append((r, c))

        if not positions:
            continue

        # Calculate centroid
        avg_row = sum(r for r, c in positions) / len(positions)
        avg_col = sum(c for r, c in positions) / len(positions)

        # Determine projection based on centroid position
        if avg_row < 5:
            # Project vertically to top and bottom borders
            col_idx = round(avg_col)
            result[0][col_idx] = color
            result[rows - 1][col_idx] = 0
        else:
            # Project horizontally to left and/or right borders
            row_idx = round(avg_row)
            if avg_col < 5.5:
                # Shape is on left side - project to left border
                result[row_idx][0] = color
                result[row_idx][cols - 1] = 0
            else:
                # Shape is on right side - project to right border
                result[row_idx][0] = 0
                result[row_idx][cols - 1] = color

    return result
