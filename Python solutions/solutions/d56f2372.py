def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large grid with multiple colored regions (different non-zero values)
    2. Output is a smaller grid containing the bounding box of one specific colored region
    3. Selection criterion depends on the number of distinct colors
    4. If 4 colors: select component with largest bounding box area
    5. Otherwise: select topmost-leftmost component (by min_row then min_col)

    Procedure:
    1. Group all cells by color
    2. For each color, calculate bounding box properties (min/max row/col, area)
    3. Count distinct colors and apply appropriate selection rule
    4. Extract the bounding box of the selected color
    """

    rows, cols = len(grid), len(grid[0])

    # Group all cells by color
    color_regions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_regions:
                    color_regions[color] = []
                color_regions[color].append((r, c))

    assert len(color_regions) > 0, "Should have at least one color region"

    # Calculate properties for each color
    color_properties = {}
    for color, cells in color_regions.items():
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        cell_count = len(cells)

        color_properties[color] = {
            "min_r": min_r,
            "max_r": max_r,
            "min_c": min_c,
            "max_c": max_c,
            "bbox_area": bbox_area,
            "cell_count": cell_count,
            "density": cell_count / bbox_area if bbox_area > 0 else 0,
        }

    # Selection criterion depends on number of colors
    num_colors = len(color_properties)

    if num_colors == 4:
        # Select color with largest bbox area
        selected_color = max(
            color_properties.keys(),
            key=lambda c: color_properties[c]["bbox_area"],
        )
    else:
        # Select topmost-leftmost component
        selected_color = min(
            color_properties.keys(),
            key=lambda c: (color_properties[c]["min_r"], color_properties[c]["min_c"]),
        )

    selected_cells = color_regions[selected_color]

    # Find bounding box of all cells of the selected color
    props = color_properties[selected_color]
    min_r, max_r = props["min_r"], props["max_r"]
    min_c, max_c = props["min_c"], props["max_c"]

    # Extract the bounding box
    result = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        result.append(row)

    return result
