def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has main canvas (large rect of 2s/0s)
    2. Small isolated patterns exist outside canvas
    3. Each hole cluster gets matched to nearest pattern
    4. Pattern is overlaid at hole's top-left, applying ALL non-zero values from pattern
    5. Key insight: overlay non-2 values only, keeping 2s from filled canvas

    Procedure:
    1. Find and extract main canvas, strip 0-borders
    2. Fill all 0s with 2s to get base output
    3. Find isolated pattern regions (connected components outside canvas)
    4. Find hole clusters in original canvas
    5. Match each cluster to nearest pattern, overlay non-2 pattern values
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])

    # Find largest rectangular region of 2s and 0s
    def find_main_canvas():
        best_region = None
        best_score = 0

        for r1 in range(rows):
            for c1 in range(cols):
                if grid[r1][c1] not in [0, 2]:
                    continue
                for r2 in range(r1, rows):
                    for c2 in range(c1, cols):
                        # Check if entire region is 0s and 2s
                        valid = True
                        count_2 = 0
                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                if grid[r][c] not in [0, 2]:
                                    valid = False
                                    break
                                if grid[r][c] == 2:
                                    count_2 += 1
                            if not valid:
                                break

                        if not valid:
                            continue

                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area < 25:  # Skip small regions
                            continue

                        score = count_2  # Prioritize regions with more 2s
                        if score > best_score:
                            best_score = score
                            best_region = (r1, c1, r2, c2)

        return best_region

    canvas_bbox = find_main_canvas()
    if not canvas_bbox:
        return []

    r1, c1, r2, c2 = canvas_bbox

    # Strip outer border of 0s from canvas
    # Top border
    while r1 <= r2 and all(grid[r1][c] == 0 for c in range(c1, c2 + 1)):
        r1 += 1
    # Bottom border
    while r1 <= r2 and all(grid[r2][c] == 0 for c in range(c1, c2 + 1)):
        r2 -= 1
    # Left border
    while c1 <= c2 and all(grid[r][c1] == 0 for r in range(r1, r2 + 1)):
        c1 += 1
    # Right border
    while c1 <= c2 and all(grid[r][c2] == 0 for r in range(r1, r2 + 1)):
        c2 -= 1

    if r1 > r2 or c1 > c2:
        return []

    canvas_h = r2 - r1 + 1
    canvas_w = c2 - c1 + 1

    # Extract and fill canvas
    result = []
    for r in range(r1, r2 + 1):
        row = []
        for c in range(c1, c2 + 1):
            val = grid[r][c]
            row.append(2 if val == 0 else val)
        result.append(row)

    # Find all patterns (regions outside canvas with special colors)
    def get_connected_component(start_r, start_c, visited):
        """Get connected component of non-zero cells"""
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr][nc] and grid[nr][nc] != 0:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.append((nr, nc))

        return cells

    visited = [[False] * cols for _ in range(rows)]
    patterns = []

    # Mark canvas as visited
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            visited[r][c] = True

    # Find all components outside canvas
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                component = get_connected_component(r, c, visited)

                # Check if component has special colors (not all 2s)
                has_special = any(grid[rr][cc] != 2 for rr, cc in component)
                if has_special:
                    # Get bounding box
                    min_r = min(rr for rr, cc in component)
                    max_r = max(rr for rr, cc in component)
                    min_c = min(cc for rr, cc in component)
                    max_c = max(cc for rr, cc in component)

                    # Extract pattern
                    pattern = []
                    for pr in range(min_r, max_r + 1):
                        row = []
                        for pc in range(min_c, max_c + 1):
                            row.append(grid[pr][pc])
                        pattern.append(row)

                    patterns.append({
                        'pattern': pattern,
                        'bbox': (min_r, min_c, max_r, max_c),
                        'cells': component
                    })

    # Find holes in canvas
    holes_visited = [[False] * canvas_w for _ in range(canvas_h)]
    hole_clusters = []

    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            canvas_r, canvas_c = r - r1, c - c1
            if grid[r][c] == 0 and not holes_visited[canvas_r][canvas_c]:
                queue = deque([(canvas_r, canvas_c)])
                holes_visited[canvas_r][canvas_c] = True
                hole_cells = [(canvas_r, canvas_c)]

                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < canvas_h and 0 <= nc < canvas_w:
                            if not holes_visited[nr][nc] and grid[r1 + nr][c1 + nc] == 0:
                                holes_visited[nr][nc] = True
                                queue.append((nr, nc))
                                hole_cells.append((nr, nc))

                hole_clusters.append(hole_cells)

    # Convert hole clusters to bbox list with center
    hole_bboxes = []
    for hole_cells in hole_clusters:
        min_hr = min(hr for hr, hc in hole_cells)
        max_hr = max(hr for hr, hc in hole_cells)
        min_hc = min(hc for hr, hc in hole_cells)
        max_hc = max(hc for hr, hc in hole_cells)
        center_r = (min_hr + max_hr) / 2
        center_c = (min_hc + max_hc) / 2
        hole_bboxes.append({
            'bbox': (min_hr, min_hc, max_hr, max_hc),
            'center': (center_r, center_c),
            'abs_center': (r1 + center_r, c1 + center_c)  # Absolute grid position
        })

    # Sort patterns and holes by position (row, then col)
    patterns_sorted = sorted(patterns, key=lambda p: (p['bbox'][0], p['bbox'][1]))
    holes_sorted = sorted(hole_bboxes, key=lambda h: (h['center'][0], h['center'][1]))

    # Create a set of hole positions for quick lookup
    all_hole_positions = set()
    for hole_cells in hole_clusters:
        for hr, hc in hole_cells:
            all_hole_positions.add((hr, hc))

    # Match patterns to holes in order
    for i, p in enumerate(patterns_sorted):
        if i < len(holes_sorted):
            hole = holes_sorted[i]
            min_hr, min_hc, max_hr, max_hc = hole['bbox']
            pattern = p['pattern']
            pat_h, pat_w = len(pattern), len(pattern[0])

            # Place pattern at top-left of hole bbox
            for pr in range(pat_h):
                for pc in range(pat_w):
                    out_r = min_hr + pr
                    out_c = min_hc + pc
                    if 0 <= out_r < canvas_h and 0 <= out_c < canvas_w:
                        # Only overlay where there's a hole OR pattern has non-2 value
                        if pattern[pr][pc] != 0:
                            if (out_r, out_c) in all_hole_positions or pattern[pr][pc] != 2:
                                result[out_r][out_c] = pattern[pr][pc]

    return result
