def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background (7) and various colored regions
    2. Output fills the bounding box of each colored region completely
    3. When bounding boxes overlap, higher-valued color takes precedence

    Procedure:
    1. Find bounding box for each non-background color
    2. Fill bounding boxes in order from lowest to highest value
    3. Higher values override lower values in overlapping areas
    """

    rows, cols = len(grid), len(grid[0])
    result = [[7] * cols for _ in range(rows)]

    # Find all non-background colors and their bounding boxes
    colors = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7:
                color = grid[r][c]
                if color not in colors:
                    colors[color] = {'min_r': r, 'max_r': r, 'min_c': c, 'max_c': c}
                else:
                    colors[color]['min_r'] = min(colors[color]['min_r'], r)
                    colors[color]['max_r'] = max(colors[color]['max_r'], r)
                    colors[color]['min_c'] = min(colors[color]['min_c'], c)
                    colors[color]['max_c'] = max(colors[color]['max_c'], c)

    # Sort by color value (lowest first, so higher values override)
    sorted_colors = sorted(colors.items(), key=lambda x: x[0])

    # Fill bounding boxes
    for color, bbox in sorted_colors:
        for r in range(bbox['min_r'], bbox['max_r'] + 1):
            for c in range(bbox['min_c'], bbox['max_c'] + 1):
                result[r][c] = color

    return result
