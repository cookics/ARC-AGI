def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10×10 grid with multiple colored regions (non-zero values) on a background of 0s
    2. Output is a 2D grid representing the bounding box of one selected colored region
    3. Each color forms a region (all cells with same non-zero value, not necessarily connected)
    4. The selected region is the one with highest density (fill ratio = cells / bbox_area)
    5. Example 1: 4s have 100% density (perfect 2×2 rectangle), 8s have 75%, 2s have 70% → output 4s
    6. Example 2: 6s have 75% density, 7s have 66.7%, 2s have 62.5% → output 6s
    7. Example 3: 5s have 75% density, 3s have 66.7%, 8s have 64.3% → output 5s

    Procedure:
    1. Group all cells by color (ignore 0s, treat each non-zero value as a region)
    2. For each colored region, compute its bounding box and density
    3. Select the region with highest density (cells / bbox_area)
    4. Extract the bounding box of the selected region, preserving 0s where the region has gaps
    5. Return the extracted bounding box as the output grid
    """

    # Group all cells by color
    rows, cols = len(grid), len(grid[0])
    color_regions = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_regions:
                    color_regions[color] = []
                color_regions[color].append((r, c))

    regions = list(color_regions.items())

    # Check for perfect rectangles first
    for color, cells in regions:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        if len(cells) == box_area:  # Perfect rectangle
            result = []
            for r in range(min_r, max_r + 1):
                row = []
                for c in range(min_c, max_c + 1):
                    row.append(color)
                result.append(row)
            return result

    # No perfect rectangle - need to determine selection criteria
    # Let me try: select region with highest fill ratio (density)
    best_region = None
    best_score = -1

    for color, cells in regions:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        box_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        fill_ratio = len(cells) / box_area

        if fill_ratio > best_score:
            best_score = fill_ratio
            best_region = (color, cells, min_r, max_r, min_c, max_c)

    # Return bounding box pattern of best region
    color, cells, min_r, max_r, min_c, max_c = best_region
    cell_set = set(cells)
    result = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            if (r, c) in cell_set:
                row.append(color)
            else:
                row.append(0)
        result.append(row)
    return result
