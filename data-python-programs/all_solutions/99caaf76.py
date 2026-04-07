def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains L-shaped patterns of 1s
    2. Each L has nearby colored values (non-1, non-8)
    3. Values rotate 180° and swap places with their associated L-shape
    4. The transformation swaps the L-shape with its rotated colored region

    Procedure:
    1. Find L-shapes (connected components of 1s)
    2. For each L, find closest colored region
    3. Rotate the colored region 180°
    4. Swap positions: L → colored region's location, rotated colors → L's location
    """

    rows, cols = len(grid), len(grid[0])
    result = [[8] * cols for _ in range(rows)]

    # Collect all 1s and all colored values
    ones = set()
    all_colors = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones.add((r, c))
            elif grid[r][c] != 8:
                all_colors[(r, c)] = grid[r][c]

    # Find connected components of 1s
    visited_ones = set()
    l_shapes = []

    for start in ones:
        if start in visited_ones:
            continue

        component = set([start])
        queue = [start]
        visited_ones.add(start)

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in ones and (nr, nc) not in visited_ones:
                    visited_ones.add((nr, nc))
                    queue.append((nr, nc))
                    component.add((nr, nc))

        # Accept any component with at least 1 cell
        if len(component) >= 1:
            l_shapes.append(component)

    # Process each L-shape
    used_colors = set()

    for l_shape in l_shapes:
        # Get L bbox
        l_rows = [r for r, c in l_shape]
        l_cols = [c for r, c in l_shape]
        l_min_r, l_max_r = min(l_rows), max(l_rows)
        l_min_c, l_max_c = min(l_cols), max(l_cols)

        # Find colored values that belong to this L
        # They should be in a bounding box adjacent to or overlapping with the L
        colored_region = {}

        # Search in all cells and find colors that share row or column range with L
        for (r, c), val in all_colors.items():
            if (r, c) in used_colors:
                continue

            # Check if color cell overlaps or is adjacent to L's row/column range
            row_overlap = l_min_r <= r <= l_max_r
            col_overlap = l_min_c <= c <= l_max_c

            if row_overlap or col_overlap:
                colored_region[(r, c)] = val

        if not colored_region:
            # No colors for this L, skip
            continue

        # Mark these colors as used
        for pos in colored_region:
            used_colors.add(pos)

        # Get color bbox
        c_rows = [r for r, c in colored_region.keys()]
        c_cols = [c for r, c in colored_region.keys()]
        c_min_r, c_max_r = min(c_rows), max(c_rows)
        c_min_c, c_max_c = min(c_cols), max(c_cols)

        # Rotate colored region 180°
        rotated_colors = {}
        for (r, c), val in colored_region.items():
            new_r = c_min_r + (c_max_r - r)
            new_c = c_min_c + (c_max_c - c)
            rotated_colors[(new_r, new_c)] = val

        # SWAP 1: Place L at colored region's position
        for r, c in l_shape:
            offset_r = r - l_min_r
            offset_c = c - l_min_c
            target_r = c_min_r + offset_r
            target_c = c_min_c + offset_c
            if 0 <= target_r < rows and 0 <= target_c < cols:
                result[target_r][target_c] = 1

        # SWAP 2: Place rotated colors at L's position
        for (r, c), val in rotated_colors.items():
            offset_r = r - c_min_r
            offset_c = c - c_min_c
            target_r = l_min_r + offset_r
            target_c = l_min_c + offset_c
            if 0 <= target_r < rows and 0 <= target_c < cols:
                result[target_r][target_c] = val

    return result
