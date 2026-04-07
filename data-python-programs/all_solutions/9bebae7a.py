def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains two colored shapes: one with color 4 (main shape) and one with color 6 (indicator)
    2. Output shows the color 4 shape reflected/mirrored and placed adjacent to the original
    3. Color 6 cells are removed in the output
    4. The reflection direction is determined by the 6-shape's bounding box aspect ratio:
       - If 6-bbox width > height: vertical reflection (top-bottom mirror)
       - Otherwise: horizontal reflection (left-right mirror)

    Procedure:
    1. Find all cells with color 4 and color 6
    2. Determine bounding boxes for both shapes
    3. Determine reflection direction based on 6-bbox aspect ratio
    4. Extract the 4-pattern within its bounding box
    5. Create a reflected copy of the pattern
    6. Place the reflection adjacent to the original (right for horizontal, below for vertical)
    7. Remove all 6-colored cells (set to 0)
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find cells with color 4 and color 6
    color_4_cells = []
    color_6_cells = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                color_4_cells.append((r, c))
            elif grid[r][c] == 6:
                color_6_cells.append((r, c))

    # Get bounding boxes
    if not color_4_cells or not color_6_cells:
        return grid

    min_r4 = min(r for r, c in color_4_cells)
    max_r4 = max(r for r, c in color_4_cells)
    min_c4 = min(c for r, c in color_4_cells)
    max_c4 = max(c for r, c in color_4_cells)

    min_r6 = min(r for r, c in color_6_cells)
    max_r6 = max(r for r, c in color_6_cells)
    min_c6 = min(c for r, c in color_6_cells)
    max_c6 = max(c for r, c in color_6_cells)

    # Determine reflection direction based on 6-bbox aspect ratio
    height_6 = max_r6 - min_r6 + 1
    width_6 = max_c6 - min_c6 + 1

    is_vertical_reflection = width_6 > height_6

    # Extract the 4-pattern within its bounding box
    height_4 = max_r4 - min_r4 + 1
    width_4 = max_c4 - min_c4 + 1

    pattern = [[grid[min_r4 + r][min_c4 + c] for c in range(width_4)] for r in range(height_4)]

    # Create output grid (start with all zeros)
    result = [[0] * cols for _ in range(rows)]

    # Place the original pattern
    for r in range(height_4):
        for c in range(width_4):
            result[min_r4 + r][min_c4 + c] = pattern[r][c]

    # Create and place the reflected pattern
    if is_vertical_reflection:
        # Vertical reflection (top-bottom flip)
        reflected = [[pattern[height_4 - 1 - r][c] for c in range(width_4)] for r in range(height_4)]

        # Determine direction: if 4-shape is above grid center, reflect downward; else upward
        grid_center_row = (rows - 1) / 2.0
        shape_center_row = (min_r4 + max_r4) / 2.0

        if shape_center_row < grid_center_row:
            # Place below the original
            start_row = max_r4 + 1
            for r in range(height_4):
                for c in range(width_4):
                    if start_row + r < rows:
                        result[start_row + r][min_c4 + c] = reflected[r][c]
        else:
            # Place above the original
            start_row = min_r4 - height_4
            for r in range(height_4):
                for c in range(width_4):
                    if start_row + r >= 0:
                        result[start_row + r][min_c4 + c] = reflected[r][c]
    else:
        # Horizontal reflection (left-right flip)
        reflected = [[pattern[r][width_4 - 1 - c] for c in range(width_4)] for r in range(height_4)]

        # Place to the right of the original
        start_col = max_c4 + 1
        for r in range(height_4):
            for c in range(width_4):
                if start_col + c < cols:
                    result[min_r4 + r][start_col + c] = reflected[r][c]

    return result
