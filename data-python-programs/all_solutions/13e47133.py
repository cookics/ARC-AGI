def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Separator values (like 2, 6, 8) form lines that divide the grid into regions
    2. Each region is filled with concentric layers based on distance from edges
    3. Pattern determined by marker values at each distance level

    Procedure:
    1. Find separator (value forming longest lines)
    2. Flood-fill to identify regions
    3. For each region, compute BFS distance from boundary
    4. Build pattern from markers sorted by their minimum distance
    5. Fill region with cycling pattern
    """
    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find separator: value with longest line
    all_counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    background = all_counts.most_common(1)[0][0]

    best_sep, best_score = None, 0
    for val in all_counts:
        if val == background:
            continue
        score = max(
            max((sum(1 for c in range(cols) if grid[r][c] == val) for r in range(rows)), default=0),
            max((sum(1 for r in range(rows) if grid[r][c] == val) for c in range(cols)), default=0)
        )
        if score > best_score:
            best_score, best_sep = score, val

    if not best_sep:
        return result

    # Flood fill regions
    visited = [[False] * cols for _ in range(rows)]
    regions = []

    for sr in range(rows):
        for sc in range(cols):
            if visited[sr][sc] or grid[sr][sc] == best_sep:
                continue

            region = []
            queue = [(sr, sc)]
            visited[sr][sc] = True

            while queue:
                r, c = queue.pop(0)
                region.append((r, c))
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != best_sep:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

            regions.append(region)

    # Process each region
    for region in regions:
        region_set = set(region)

        # BFS for distances from boundary
        dist = {}
        queue = deque()

        for r, c in region:
            is_edge = any((r+dr, c+dc) not in region_set for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)])
            if is_edge:
                dist[(r,c)] = 0
                queue.append((r,c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in region_set and (nr, nc) not in dist:
                    dist[(nr,nc)] = dist[(r,c)] + 1
                    queue.append((nr,nc))

        # Find markers by minimum distance
        val_min_dist = {}
        for r, c in region:
            v = grid[r][c]
            d = dist[(r,c)]
            if v not in val_min_dist or d < val_min_dist[v]:
                val_min_dist[v] = d

        # Determine pattern
        val_counts = Counter(grid[r][c] for r, c in region)
        reg_bg = val_counts.most_common(1)[0][0]
        markers = {v: d for v, d in val_min_dist.items() if v != reg_bg}

        if not markers:
            pattern = [reg_bg]
        elif len(markers) == 1:
            mv, md = list(markers.items())[0]
            pattern = [mv] if md == 0 else [reg_bg, mv]
        else:
            pattern = [v for v, d in sorted(markers.items(), key=lambda x: (x[1], x[0]))]

        # Fill
        for r, c in region:
            result[r][c] = pattern[dist[(r,c)] % len(pattern)]

    return result
