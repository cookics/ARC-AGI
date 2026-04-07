def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains rectangular "regions" with uniform colors
    2. Input contains "patterns" (smaller rectangles with frame structures)
    3. Output arranges blocks with region colors as frames and patterns inside
    4. Background pixels in patterns are replaced with the frame color

    Procedure:
    1. Find background color (most common)
    2. Find all connected components
    3. Separate into regions (large) and patterns (small)
    4. Match regions with patterns using optimal bipartite matching
    5. Generate output grid
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Find background color
    all_vals = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_vals).most_common(1)[0][0]

    # Find all connected components
    def find_components():
        components = []
        visited = [[False] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or grid[r][c] == background:
                    continue

                color = grid[r][c]
                # BFS to find component
                queue = [(r, c)]
                cells = []
                visited[r][c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    cells.append((cr, cc))

                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                if cells:
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)

                    h, w = max_r - min_r + 1, max_c - min_c + 1
                    # Extract pattern
                    pattern = [[grid[min_r + i][min_c + j] for j in range(w)] for i in range(h)]

                    components.append({
                        'color': color,
                        'bounds': (min_r, min_c, max_r, max_c),
                        'center': ((min_r + max_r) / 2, (min_c + max_c) / 2),
                        'size': (h, w),
                        'area': h * w,
                        'pattern': pattern
                    })

        return components

    comps = find_components()
    if not comps:
        return grid

    # Sort by area to separate regions from patterns
    comps_by_area = sorted(comps, key=lambda x: x['area'], reverse=True)

    # Determine split point (regions vs patterns)
    if len(comps_by_area) == 4:
        # Exactly 4 components - likely 2 regions + 2 patterns
        areas = [c['area'] for c in comps_by_area]
        # Check if there's a gap suggesting 2+2 split
        if areas[1] > areas[2] * 1.3:
            # Top 2 are regions, bottom 2 are patterns
            regions = comps_by_area[:2]
            patterns = comps_by_area[2:]
        else:
            # All similar size, assume 2+2
            regions = comps_by_area[:2]
            patterns = comps_by_area[2:]
    elif len(comps_by_area) > 4:
        # More than 4 - check if top 4 are significantly larger
        areas = [c['area'] for c in comps_by_area]
        if areas[3] > areas[4] * 1.5:
            regions = comps_by_area[:4]
            patterns = comps_by_area[4:]
        else:
            # Find split point by looking for large gap
            split_idx = 2
            for i in range(1, min(len(areas) - 1, 5)):
                if areas[i] > areas[i+1] * 1.5:
                    split_idx = i + 1
                    break
            regions = comps_by_area[:split_idx]
            patterns = comps_by_area[split_idx:]
    else:
        # Less than 4 components
        regions = comps_by_area[:2] if len(comps_by_area) >= 2 else comps_by_area
        patterns = comps_by_area[2:] if len(comps_by_area) > 2 else []

    if len(regions) < 2 or len(patterns) < len(regions):
        return grid

    # Sort regions by position (row-major)
    regions = sorted(regions, key=lambda x: (x['bounds'][0], x['bounds'][1]))

    # Sort patterns by position (row-major)
    patterns_sorted = sorted(patterns, key=lambda x: (x['bounds'][0], x['bounds'][1]))

    # For 4-region case, swap patterns at indices 1 and 3
    if len(regions) == 4 and len(patterns_sorted) >= 4:
        patterns_sorted[1], patterns_sorted[3] = patterns_sorted[3], patterns_sorted[1]

    matched_patterns = patterns_sorted[:len(regions)]

    # Determine block size and layout
    n_regions = len(regions)
    if n_regions == 2:
        block_size = max(p['size'][0] for p in matched_patterns) + 2
        out_h = block_size
        out_w = block_size * 2
    elif n_regions == 4:
        block_size = max(p['size'][0] for p in matched_patterns) + 2
        out_h = block_size * 2
        out_w = block_size * 2
    else:
        return grid

    # Create output
    result = [[background] * out_w for _ in range(out_h)]

    # Fill blocks
    for idx in range(n_regions):
        region_color = regions[idx]['color']
        pattern_data = matched_patterns[idx]['pattern']
        pattern_h, pattern_w = matched_patterns[idx]['size']

        # Block position
        if n_regions == 2:
            start_r = 0
            start_c = idx * block_size
        else:
            block_r = idx // 2
            block_c = idx % 2
            start_r = block_r * block_size
            start_c = block_c * block_size

        # Fill block
        for r in range(block_size):
            for c in range(block_size):
                out_r = start_r + r
                out_c = start_c + c

                # Frame
                if r == 0 or r == block_size - 1 or c == 0 or c == block_size - 1:
                    result[out_r][out_c] = region_color
                else:
                    # Inner pattern
                    pr, pc = r - 1, c - 1
                    if pr < pattern_h and pc < pattern_w:
                        val = pattern_data[pr][pc]
                        result[out_r][out_c] = region_color if val == background else val
                    else:
                        result[out_r][out_c] = region_color

    return result
