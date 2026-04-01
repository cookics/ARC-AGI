def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a variable-sized grid containing scattered 2s that form distinct spatial regions.
    2. Output is always a 7x7 grid with four 3x3 blocks arranged in a 2x2 pattern.
    3. The 3x3 blocks are separated by empty rows/columns (row 3 and column 3 are zeros).
    4. Each 3x3 block in the output corresponds to a 3x3 window extracted from around each region in the input.
    5. The spatial arrangement of regions in input determines their placement in the 2x2 output layout.

    Procedure:
    1. Find all distinct regions by clustering 2s within 3x3 neighborhoods to separate spatially distant groups.
    2. Calculate the center position of each region using its bounding box.
    3. Extract a 3x3 window centered on each region's center position.
    4. Classify regions into quadrants based on median row and column positions.
    5. Map the four region windows to their corresponding positions in the 7x7 output grid.
    """
    rows, cols = len(grid), len(grid[0])

    # Find all positions with 2s
    twos_positions = [
        (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2
    ]

    if not twos_positions:
        return [[0] * 7 for _ in range(7)]

    # Use more aggressive clustering to separate regions
    regions = []
    used = set()

    def get_connected_region(start_r, start_c):
        """Get all positions connected within immediate proximity (3x3 neighborhood)"""
        region = [(start_r, start_c)]
        used.add((start_r, start_c))
        queue = [(start_r, start_c)]

        while queue:
            r, c = queue.pop(0)
            for r2, c2 in twos_positions:
                if (r2, c2) not in used:
                    # Only connect if within 3x3 neighborhood (adjacent or diagonal)
                    if abs(r2 - r) <= 2 and abs(c2 - c) <= 2:
                        used.add((r2, c2))
                        region.append((r2, c2))
                        queue.append((r2, c2))

        return region

    # Find all distinct regions
    for r, c in twos_positions:
        if (r, c) not in used:
            region = get_connected_region(r, c)
            if region:
                # Find bounding box center
                min_r = min(rr for rr, cc in region)
                max_r = max(rr for rr, cc in region)
                min_c = min(cc for rr, cc in region)
                max_c = max(cc for rr, cc in region)
                center_r = (min_r + max_r) // 2
                center_c = (min_c + max_c) // 2
                regions.append((center_r, center_c, region))

    # Extract 3x3 windows for each region
    region_windows = []
    for center_r, center_c, region_points in regions:
        # Extract 3x3 window centered on the region
        window = []
        for r in range(center_r - 1, center_r + 2):
            row = []
            for c in range(center_c - 1, center_c + 2):
                if 0 <= r < rows and 0 <= c < cols:
                    row.append(grid[r][c])
                else:
                    row.append(0)
            window.append(row)
        region_windows.append((center_r, center_c, window))

    # Create 7x7 result grid
    result = [[0 for _ in range(7)] for _ in range(7)]

    if len(region_windows) >= 4:
        # Determine relative positions and map to 2x2 grid
        # Find the median row and column to split regions
        rows_list = [r for r, c, w in region_windows]
        cols_list = [c for r, c, w in region_windows]
        median_row = sorted(rows_list)[len(rows_list) // 2]
        median_col = sorted(cols_list)[len(cols_list) // 2]

        # Classify each region as top/bottom and left/right
        top_regions = [(r, c, w) for r, c, w in region_windows if r < median_row]
        bottom_regions = [(r, c, w) for r, c, w in region_windows if r >= median_row]

        # Sort within each group by column
        top_regions.sort(key=lambda x: x[1])
        bottom_regions.sort(key=lambda x: x[1])

        # Map to output positions
        if len(top_regions) >= 2:
            # Top-left
            if len(top_regions) > 0:
                _, _, window = top_regions[0]
                for r in range(3):
                    for c in range(3):
                        result[r][c] = window[r][c]
            # Top-right
            if len(top_regions) > 1:
                _, _, window = top_regions[1]
                for r in range(3):
                    for c in range(3):
                        result[r][c + 4] = window[r][c]

        if len(bottom_regions) >= 2:
            # Bottom-left
            if len(bottom_regions) > 0:
                _, _, window = bottom_regions[0]
                for r in range(3):
                    for c in range(3):
                        result[r + 4][c] = window[r][c]
            # Bottom-right
            if len(bottom_regions) > 1:
                _, _, window = bottom_regions[1]
                for r in range(3):
                    for c in range(3):
                        result[r + 4][c + 4] = window[r][c]
    else:
        # Handle fewer than 4 regions - place sequentially
        positions = [(0, 0), (0, 4), (4, 0), (4, 4)]
        for i, (_, _, window) in enumerate(region_windows):
            if i < len(positions):
                start_r, start_c = positions[i]
                for r in range(3):
                    for c in range(3):
                        result[start_r + r][start_c + c] = window[r][c]

    return result
