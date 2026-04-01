def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has 8s forming lattice structure with rectangular cells
    2. Colored pattern indicates what should fill gaps and connected empty cells
    3. Gaps are positions on grid lines where 8s are missing

    Procedure:
    1. Find colored value
    2. Identify dominant grid lines (rows/cols with >50% 8s)
    3. Find all gaps (0s on dominant lines) within the region containing colored cells
    4. BFS from colored cells AND gaps to fill connected 0s
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find color
    color = None
    colored_cells = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] not in [0, 8]:
                color = grid[i][j]
                colored_cells.append((i, j))

    if not color:
        return [row[:] for row in grid]

    # Find dominant grid lines
    col_8_pct = [sum(1 for i in range(rows) if grid[i][j] == 8) / rows for j in range(cols)]
    row_8_pct = [sum(1 for j in range(cols) if grid[i][j] == 8) / cols for i in range(rows)]

    dominant_cols = [j for j in range(cols) if col_8_pct[j] > 0.5]
    dominant_rows = [i for i in range(rows) if row_8_pct[i] > 0.5]

    # Find bounding region: all dominant rows/cols that span the colored cells
    min_r = min(i for i, j in colored_cells)
    max_r = max(i for i, j in colored_cells)
    min_c = min(j for i, j in colored_cells)
    max_c = max(j for i, j in colored_cells)

    # Expand to include all intermediate dominant lines
    active_rows = [r for r in dominant_rows if min_r <= r <= max_r]
    active_cols = [c for c in dominant_cols if min_c <= c <= max_c]

    # Add boundary dominant lines
    if dominant_rows:
        for r in dominant_rows:
            if r < min_r:
                min_r = r
                break
        for r in reversed(dominant_rows):
            if r > max_r:
                max_r = r
                break
    if dominant_cols:
        for c in dominant_cols:
            if c < min_c:
                min_c = c
                break
        for c in reversed(dominant_cols):
            if c > max_c:
                max_c = c
                break

    # Find gaps: 0s on dominant lines within bounding region
    gaps = []
    for j in dominant_cols:
        if min_c <= j <= max_c:
            for i in range(min_r, max_r + 1):
                if grid[i][j] == 0:
                    gaps.append((i, j))
    for i in dominant_rows:
        if min_r <= i <= max_r:
            for j in range(min_c, max_c + 1):
                if grid[i][j] == 0 and (i, j) not in gaps:
                    gaps.append((i, j))

    # Result grid
    result = [row[:] for row in grid]

    # Fill gaps
    for i, j in gaps:
        result[i][j] = color

    # BFS from colored cells and gaps
    visited = set(colored_cells + gaps)
    queue = colored_cells + gaps

    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if min_r <= nr <= max_r and min_c <= nc <= max_c:
                if (nr, nc) not in visited and grid[nr][nc] == 0:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    result[nr][nc] = color

    return result
