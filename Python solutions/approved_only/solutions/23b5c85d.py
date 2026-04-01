def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with 0 representing background and non-zero numbers representing different colored regions.
    2. Output is the rectangular region with the smallest area among all non-zero colored regions.
    3. The output region should be returned as a solid block filled with the same color value.

    Procedure:
    1. Identify all unique non-zero colors in the grid.
    2. For each color, find its bounding box (minimum and maximum row/column indices).
    3. Calculate the area of each color's bounding box.
    4. Identify the color with the smallest bounding box area.
    5. Return a rectangular grid filled with that color, with dimensions matching its bounding box.
    """

    rows, cols = len(grid), len(grid[0])

    # Find all unique non-zero colors
    colors = set()
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                colors.add(grid[i][j])

    smallest_area = float("inf")
    smallest_region = None

    # For each color, find its bounding box and calculate area
    for color in colors:
        min_row, max_row = rows, -1
        min_col, max_col = cols, -1

        # Find bounding box for this color
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == color:
                    min_row = min(min_row, i)
                    max_row = max(max_row, i)
                    min_col = min(min_col, j)
                    max_col = max(max_col, j)

        if max_row >= 0:  # Found at least one cell of this color
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            area = height * width

            if area < smallest_area:
                smallest_area = area
                smallest_region = [[color] * width for _ in range(height)]

    return smallest_region
