def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains scattered rectangular patches (corners/edges)
    2. Patches should be arranged in 2x2 grid WITH 1-cell overlap on borders
    3. Border cells overlap when assembling
    4. Result is centered

    Procedure:
    1. Find background color
    2. Extract connected components as patches
    3. Classify into quadrants (TL, TR, BL, BR)
    4. Merge patches within each quadrant
    5. Assemble in 2x2 grid with 1-cell overlap
    6. Center result
    """

    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])

    # Find background color
    flat = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(flat).most_common(1)[0][0]

    # Find connected components
    non_bg_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != bg:
                non_bg_cells.add((r, c))

    if not non_bg_cells:
        return [[bg] * cols for _ in range(rows)]

    visited = set()
    patches = []

    def bfs(start_r, start_c):
        q = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        component = [(start_r, start_c)]

        while q:
            r, c = q.popleft()
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in non_bg_cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc))
                    component.append((nr, nc))
        return component

    for r, c in non_bg_cells:
        if (r, c) not in visited:
            patches.append(bfs(r, c))

    if not patches:
        return [[bg] * cols for _ in range(rows)]

    # Overlay ALL patches at (0,0) - normalize each to origin
    merged = {}
    for patch in patches:
        min_r = min(r for r, c in patch)
        min_c = min(c for r, c in patch)

        for r, c in patch:
            norm_r, norm_c = r - min_r, c - min_c
            color = grid[r][c]

            if color != bg:
                # First write wins
                if (norm_r, norm_c) not in merged:
                    merged[(norm_r, norm_c)] = color

    if not merged:
        return [[bg] * cols for _ in range(rows)]

    # Build final grid
    rs = [r for r, c in merged.keys()]
    cs = [c for r, c in merged.keys()]
    final_h = max(rs) - min(rs) + 1
    final_w = max(cs) - min(cs) + 1

    final = [[bg] * final_w for _ in range(final_h)]
    for (r, c), color in merged.items():
        final[r - min(rs)][c - min(cs)] = color

    # Center in output
    result = [[bg] * cols for _ in range(rows)]
    start_r = (rows - final_h) // 2
    start_c = (cols - final_w) // 2

    for r in range(final_h):
        for c in range(final_w):
            if 0 <= start_r + r < rows and 0 <= start_c + c < cols:
                result[start_r + r][start_c + c] = final[r][c]

    return result
