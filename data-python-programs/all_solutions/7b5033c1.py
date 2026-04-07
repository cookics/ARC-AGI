def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a background color and several objects in different colors
    2. Output is a single column listing non-background colors
    3. Each color appears as many times as it occurs in the input grid
    4. Colors are ordered by their first appearance (top-to-bottom, left-to-right scan)
    5. Background color is the most frequent color in the grid

    Procedure:
    1. Count occurrences of each color in the grid
    2. Identify background color as the most frequent
    3. Track first appearance position for each non-background color
    4. Sort non-background colors by their first appearance
    5. Build output by repeating each color according to its count
    """

    # Flatten grid and count colors
    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1

    # Find background color (most frequent)
    background_color = max(color_counts, key=color_counts.get)

    # Find first appearance of each non-background color
    color_first_appearance = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell != background_color and cell not in color_first_appearance:
                color_first_appearance[cell] = (r, c)

    # Sort colors by first appearance
    sorted_colors = sorted(
        color_first_appearance.keys(), key=lambda x: color_first_appearance[x]
    )

    # Build output
    result = []
    for color in sorted_colors:
        count = color_counts[color]
        for _ in range(count):
            result.append([color])

    return result
