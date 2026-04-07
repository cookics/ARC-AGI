def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with disconnected colored regions (non-zero values)
    2. Output is a 2D grid where each column represents one color
    3. The height of each column (number of non-zero entries) equals the number of distinct rows that color appears in
    4. Columns are ordered by row count (descending), then by color value (ascending) as tiebreaker

    Procedure:
    1. Iterate through input grid and track which rows each non-zero color appears in
    2. Count the number of distinct rows for each color
    3. Sort colors by row count (descending), then by color value (ascending)
    4. Create output grid with height = max row count and width = number of colors
    5. Fill each column with its color for the first (row count) rows, then 0s
    """

    # Find all non-zero colors and their row spans
    color_heights = {}

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] != 0:
                color = grid[row][col]
                if color not in color_heights:
                    color_heights[color] = set()
                color_heights[color].add(row)

    # Calculate height for each color (number of distinct rows it appears in)
    color_height_list = []
    for color, rows in color_heights.items():
        height = len(rows)
        color_height_list.append((height, color))

    # Sort by height (descending), then by color (for stable ordering)
    color_height_list.sort(key=lambda x: (-x[0], x[1]))

    # Create output grid
    if not color_height_list:
        return []

    max_height = color_height_list[0][0]
    num_colors = len(color_height_list)

    result = []
    for row in range(max_height):
        result_row = []
        for col in range(num_colors):
            height, color = color_height_list[col]
            if row < height:
                result_row.append(color)
            else:
                result_row.append(0)
        result.append(result_row)

    return result
