def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid contains a "background" value (most common, like 9, 0, or 8)
    2. A "separator" value forms rectangular frame structures
    3. Small "patches" of other values appear in corners/regions
    4. Separator cells get replaced based on closest patch
    5. Rectangular frame borders are preserved as separator

    Procedure:
    1. Identify background (most common) and separator values
    2. Find all hollow rectangular frames formed by separator
    3. Identify patches (non-background, non-separator regions)
    4. Fill separator cells based on closest patch (Euclidean distance)
    5. Keep rectangle borders as separator
    """
    from collections import Counter, deque
    import math

    n, m = len(grid), len(grid[0])

    # Count all values to identify background
    all_vals = [grid[i][j] for i in range(n) for j in range(m)]
    counter = Counter(all_vals)

    # Background is most common
    background = counter.most_common(1)[0][0]

    # Try candidates for separator value
    candidates = [val for val, _ in counter.most_common(10) if val != background]

    def find_hollow_rectangles(sep_val):
        """Find all rectangles with sep_val on borders"""
        rects = []
        for r1 in range(n):
            for c1 in range(m):
                for r2 in range(r1+2, n):
                    for c2 in range(c1+2, m):
                        # Check if border is all sep_val
                        is_rect = True
                        # Check top and bottom rows
                        for c in range(c1, c2+1):
                            if grid[r1][c] != sep_val or grid[r2][c] != sep_val:
                                is_rect = False
                                break
                        # Check left and right columns
                        if is_rect:
                            for r in range(r1, r2+1):
                                if grid[r][c1] != sep_val or grid[r][c2] != sep_val:
                                    is_rect = False
                                    break

                        if is_rect:
                            rects.append((r1, c1, r2, c2))
        return rects

    # Find the best separator with valid rectangles
    best_sep = None
    best_rects = []
    for sep in candidates:
        rects = find_hollow_rectangles(sep)
        if rects:
            best_sep = sep
            best_rects = rects
            break

    if not best_rects:
        return grid  # No frame found, return as-is

    # Find all patches (connected components of non-background, non-separator values)
    visited = [[False] * m for _ in range(n)]
    patches = []

    def bfs_patch(start_r, start_c):
        """Find connected component using BFS"""
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]
        vals = [grid[start_r][start_c]]

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m and not visited[nr][nc]:
                    val = grid[nr][nc]
                    if val != background and val != best_sep:
                        visited[nr][nc] = True
                        queue.append((nr, nc))
                        cells.append((nr, nc))
                        vals.append(val)

        return cells, vals

    # Collect all cells by value (not by connected component)
    # Exclude background and separator value
    value_cells = {}
    for i in range(n):
        for j in range(m):
            val = grid[i][j]
            if val != background and val != best_sep:
                if val not in value_cells:
                    value_cells[val] = []
                value_cells[val].append((i, j))

    # Compute average position for each value
    value_centroids = {}
    for val, cells in value_cells.items():
        avg_row = sum(r for r, c in cells) / len(cells)
        avg_col = sum(c for r, c in cells) / len(cells)
        value_centroids[val] = (avg_row, avg_col)

    # Create output grid
    result = [row[:] for row in grid]

    # Function to check if a cell is on the border of any rectangle
    def is_on_rect_border(i, j):
        for r1, c1, r2, c2 in best_rects:
            # Check if on this rectangle's border
            if (i == r1 or i == r2) and c1 <= j <= c2:
                return True
            if (j == c1 or j == c2) and r1 <= i <= r2:
                return True
        return False

    # Fill separator cells using distance to value centroids
    for i in range(n):
        for j in range(m):
            if grid[i][j] == best_sep:
                # Check if on any rectangle border - keep these as separator
                if is_on_rect_border(i, j):
                    continue

                # Find value with closest centroid
                min_dist = float('inf')
                closest_val = background

                for val, (avg_row, avg_col) in value_centroids.items():
                    dist = math.sqrt((i - avg_row)**2 + (j - avg_col)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_val = val
                    elif abs(dist - min_dist) < 1e-9 and val > closest_val:
                        # Use stricter equality check for floats
                        closest_val = val

                result[i][j] = closest_val

    return result
