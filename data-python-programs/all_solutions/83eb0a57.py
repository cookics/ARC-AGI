def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with a dominant background color
    2. There are multiple rectangular regions with different colors
    3. Output is the size of the largest non-background region
    4. Smaller regions are overlaid onto larger regions in a nested manner
    5. Regions are aligned based on special marker values (like 1s) that appear inside them

    Procedure:
    1. Identify background color (most frequent value)
    2. Find all rectangular regions that are non-background
    3. Extract each region with its bounding box
    4. Sort regions by area (largest first)
    5. Start with largest region as base output
    6. For each smaller region, find alignment based on marker values
    7. Overlay smaller regions onto the output
    """

    from collections import Counter

    def find_background(grid):
        """Find the most common value in grid"""
        flat = [val for row in grid for val in row]
        return Counter(flat).most_common(1)[0][0]

    def find_bounding_box(grid, bg_color):
        """Find bounding boxes of all non-background regions"""
        h, w = len(grid), len(grid[0])
        visited = [[False] * w for _ in range(h)]
        regions = []

        for i in range(h):
            for j in range(w):
                if not visited[i][j] and grid[i][j] != bg_color:
                    # Found a new region, find its bounding box
                    color = grid[i][j]
                    min_r, max_r = i, i
                    min_c, max_c = j, j

                    # BFS/DFS to find connected component
                    queue = [(i, j)]
                    visited[i][j] = True
                    component = [(i, j)]

                    while queue:
                        r, c = queue.pop(0)
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc]:
                                if grid[nr][nc] != bg_color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                                    component.append((nr, nc))
                                    min_r = min(min_r, nr)
                                    max_r = max(max_r, nr)
                                    min_c = min(min_c, nc)
                                    max_c = max(max_c, nc)

                    # Extract the region
                    region = []
                    for r in range(min_r, max_r + 1):
                        row = []
                        for c in range(min_c, max_c + 1):
                            row.append(grid[r][c])
                        region.append(row)

                    regions.append({
                        'data': region,
                        'bbox': (min_r, min_c, max_r, max_c),
                        'area': (max_r - min_r + 1) * (max_c - min_c + 1)
                    })

        return regions

    def find_markers(region, frame_color):
        """Find positions of non-frame values in region, separating isolated vs clustered"""
        markers = {}
        isolated_markers = {}
        h, w = len(region), len(region[0])
        for i in range(h):
            for j in range(w):
                val = region[i][j]
                if val != frame_color:
                    # Check if isolated (no adjacent cells with same value)
                    is_isolated = True
                    for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w and region[ni][nj] == val:
                            is_isolated = False
                            break

                    if val not in markers:
                        markers[val] = []
                        isolated_markers[val] = []
                    markers[val].append((i, j))
                    if is_isolated:
                        isolated_markers[val].append((i, j))
        return markers, isolated_markers

    def overlay_region(base, region, pos_r, pos_c):
        """Overlay region onto base at given position"""
        h, w = len(region), len(region[0])
        base_h, base_w = len(base), len(base[0])

        for i in range(h):
            for j in range(w):
                br, bc = pos_r + i, pos_c + j
                if 0 <= br < base_h and 0 <= bc < base_w:
                    base[br][bc] = region[i][j]

    def find_best_alignment(base, region, base_frame_color, region_frame_color):
        """Find best position to place region on base by aligning markers"""
        base_markers, base_isolated = find_markers(base, base_frame_color)
        region_markers, region_isolated = find_markers(region, region_frame_color)

        # First, try to align isolated markers (single markers should match single markers)
        for val in region_isolated:
            if val in base_isolated and len(base_isolated[val]) > 0 and len(region_isolated[val]) > 0:
                # Align first isolated marker of this value
                base_pos = base_isolated[val][0]
                region_pos = region_isolated[val][0]
                # Calculate offset
                offset_r = base_pos[0] - region_pos[0]
                offset_c = base_pos[1] - region_pos[1]
                return (offset_r, offset_c)

        # If no isolated markers, try to align on any common marker values
        for val in region_markers:
            if val in base_markers and len(base_markers[val]) > 0 and len(region_markers[val]) > 0:
                # Align first marker of this value
                base_pos = base_markers[val][0]
                region_pos = region_markers[val][0]
                # Calculate offset
                offset_r = base_pos[0] - region_pos[0]
                offset_c = base_pos[1] - region_pos[1]
                return (offset_r, offset_c)

        # Default: place at (1, 1) if no alignment found
        return (1, 1)

    # Main algorithm
    bg_color = find_background(grid)
    regions = find_bounding_box(grid, bg_color)

    # Sort by area (largest first)
    regions.sort(key=lambda x: x['area'], reverse=True)

    if not regions:
        return grid

    # Start with largest region as base
    output = [row[:] for row in regions[0]['data']]
    base_frame_color = regions[0]['data'][0][0]  # Assume frame is the border value

    # Overlay smaller regions
    for i in range(1, len(regions)):
        region = regions[i]['data']
        region_frame_color = region[0][0]

        pos_r, pos_c = find_best_alignment(output, region, base_frame_color, region_frame_color)
        overlay_region(output, region, pos_r, pos_c)

    return output
