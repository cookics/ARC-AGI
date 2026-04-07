def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Isolated cells (small components) are markers - reflect for 4-fold symmetry
    2. Shape (largest component) stays mostly in place
    3. Specific shape edge cells extend in controlled directions
    4. Center rows above/below middle extend in opposite directions

    Procedure:
    1. Find markers (isolated) and shape (largest component)
    2. Place markers at 4 symmetric positions around shape center
    3. Extend shape cells in center rows outward
    4. Copy topmost shape cells upward if marker is above
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all non-zero cells
    all_nz = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                all_nz.add((r, c))

    if not all_nz:
        return result

    # Find connected components
    visited = set()
    components = []
    for start in all_nz:
        if start in visited:
            continue
        comp = []
        q = deque([start])
        visited.add(start)
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in all_nz and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        components.append(comp)

    # Get shape (largest component)
    main_comp = max(components, key=len)
    shape_rs = [r for r, c in main_comp]
    shape_cs = [c for r, c in main_comp]
    shape_r_min, shape_r_max = min(shape_rs), max(shape_rs)
    shape_c_min, shape_c_max = min(shape_cs), max(shape_cs)

    # Shape center
    center_r = (shape_r_min + shape_r_max) / 2.0
    center_c = (shape_c_min + shape_c_max) / 2.0

    # Find all markers (non-shape components)
    markers = []
    for comp in components:
        if comp != main_comp:
            markers.extend(comp)

    # Reflect each marker to create 4-fold symmetry
    for mr, mc in markers:
        marker_val = grid[mr][mc]
        for nr, nc in [
            (mr, mc),                                                    # Original
            (mr, int(2 * center_c - mc)),                               # H-reflect
            (int(2 * center_r - mr), mc),                               # V-reflect
            (int(2 * center_r - mr), int(2 * center_c - mc))           # Both
        ]:
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = marker_val

    # Extend shape cells in rows near center
    for r, c in main_comp:
        val = grid[r][c]

        # Rows near vertical center extend horizontally
        dist_from_center = abs(r - center_r)
        if dist_from_center < 1.0:  # Within 1 row of center
            # Get row info
            row_cells = [(rr, cc) for rr, cc in main_comp if rr == r]
            if row_cells:
                row_cols = [cc for _, cc in row_cells]
                min_col = min(row_cols)
                max_col = max(row_cols)

                # Rows above center extend right
                if r < center_r:
                    if c == max_col and c < cols - 1:
                        result[r][c + 1] = val
                # Rows below center extend left
                elif r > center_r:
                    if c == min_col and c > 0:
                        result[r][c - 1] = val
                # Row at exact center extends both ways
                else:
                    if c == min_col and c > 0:
                        result[r][c - 1] = val
                    if c == max_col and c < cols - 1:
                        result[r][c + 1] = val

        # Copy topmost shape row upward if there's a marker above
        if r == shape_r_min and any(mr < shape_r_min for mr, mc in markers):
            row_cells = [(rr, cc) for rr, cc in main_comp if rr == shape_r_min]
            if row_cells:
                row_cols = [cc for _, cc in row_cells]
                min_col = min(row_cols)
                # Copy leftmost cell up
                if c == min_col and r > 0:
                    result[r - 1][c] = val

    return result
