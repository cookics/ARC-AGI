def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Patterns have fill regions surrounded by incomplete frames
    2. Rays extend from frame gaps and from fill region edges
    3. Gap at corner → diagonal ray; gap at edge → straight ray
    4. Fill regions also create vertical/horizontal rays through the entire grid

    Procedure:
    1. Find patterns (fill regions with single border color)
    2. Fill gaps and create diagonal/straight rays from gaps
    3. Create additional rays from fill region edges extending through entire grid
    """

    from collections import Counter, deque

    H, W = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    def extend_ray(r, c, dr, dc, color):
        nr, nc = r + dr, c + dc
        while 0 <= nr < H and 0 <= nc < W and result[nr][nc] == bg and grid[nr][nc] == bg:
            result[nr][nc] = color
            nr += dr
            nc += dc

    # Find components
    visited = [[False] * W for _ in range(H)]
    components = []

    for sr in range(H):
        for sc in range(W):
            if visited[sr][sc] or grid[sr][sc] == bg:
                continue
            color = grid[sr][sc]
            cells = []
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        q.append((nr, nc))
            components.append((color, cells))

    # Find patterns
    patterns = []
    for fill_color, fill_cells in components:
        if len(fill_cells) < 4:
            continue

        adj_colors = {}
        for r, c in fill_cells:
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    nbr = grid[nr][nc]
                    if nbr != bg and nbr != fill_color:
                        adj_colors[nbr] = adj_colors.get(nbr, 0) + 1

        if len(adj_colors) != 1:
            continue

        border_color = list(adj_colors.keys())[0]
        fill_set = set(fill_cells)
        border_cells = set()
        for r, c in fill_cells:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == border_color:
                        border_cells.add((nr, nc))

        if not border_cells or len(border_cells) < len(fill_cells) * 1.2:
            continue

        all_cells = list(fill_set | border_cells)
        min_r = min(r for r, c in all_cells)
        max_r = max(r for r, c in all_cells)
        min_c = min(c for r, c in all_cells)
        max_c = max(c for r, c in all_cells)

        # Get fill bbox
        fill_min_r = min(r for r, c in fill_set)
        fill_max_r = max(r for r, c in fill_set)
        fill_min_c = min(c for r, c in fill_set)
        fill_max_c = max(c for r, c in fill_set)

        patterns.append((fill_color, fill_set, min_r, max_r, min_c, max_c, fill_min_r, fill_max_r, fill_min_c, fill_max_c))

    # Process patterns: fill gaps and create rays
    for fill_color, fill_set, min_r, max_r, min_c, max_c, fill_min_r, fill_max_r, fill_min_c, fill_max_c in patterns:
        # Find and fill gaps
        gaps = []
        for c in range(min_c, max_c + 1):
            if grid[min_r][c] == bg:
                gaps.append((min_r, c))
            if grid[max_r][c] == bg:
                gaps.append((max_r, c))
        for r in range(min_r + 1, max_r):
            if grid[r][min_c] == bg:
                gaps.append((r, min_c))
            if grid[r][max_c] == bg:
                gaps.append((r, max_c))

        for r, c in gaps:
            result[r][c] = fill_color

            is_top = (r == min_r)
            is_bottom = (r == max_r)
            is_left = (c == min_c)
            is_right = (c == max_c)

            if is_top and is_left:
                extend_ray(r, c, -1, -1, fill_color)
            elif is_top and is_right:
                extend_ray(r, c, -1, 1, fill_color)
            elif is_bottom and is_left:
                extend_ray(r, c, 1, -1, fill_color)
            elif is_bottom and is_right:
                extend_ray(r, c, 1, 1, fill_color)
            elif is_top:
                extend_ray(r, c, -1, 0, fill_color)
            elif is_bottom:
                extend_ray(r, c, 1, 0, fill_color)
            elif is_left:
                extend_ray(r, c, 0, -1, fill_color)
            elif is_right:
                extend_ray(r, c, 0, 1, fill_color)

        # Additional: extend from fill edges in all 4 directions
        # Top edge of fill
        for c in range(fill_min_c, fill_max_c + 1):
            if (fill_min_r, c) in fill_set:
                extend_ray(fill_min_r, c, -1, 0, fill_color)
        # Bottom edge of fill
        for c in range(fill_min_c, fill_max_c + 1):
            if (fill_max_r, c) in fill_set:
                extend_ray(fill_max_r, c, 1, 0, fill_color)
        # Left edge of fill
        for r in range(fill_min_r, fill_max_r + 1):
            if (r, fill_min_c) in fill_set:
                extend_ray(r, fill_min_c, 0, -1, fill_color)
        # Right edge of fill
        for r in range(fill_min_r, fill_max_r + 1):
            if (r, fill_max_c) in fill_set:
                extend_ray(r, fill_max_c, 0, 1, fill_color)

    return result
