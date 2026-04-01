def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a 3×3 block of 8s and scattered 5s (markers)
    2. From the 4 corners: diagonal rays place 2s (stop at 5s/8s/boundary)
    3. From the 4 edge midpoints: perpendicular rays place 4s
    4. Additional 4s spread from perpendicular rays via BFS
    5. BFS is bounded by 2s, 5s, 8s (not geometric formulas)

    Procedure:
    1. Find the 3×3 block of 8s
    2. Shoot diagonal rays from corners (mark with 2s)
    3. Shoot perpendicular rays from edge midpoints (mark with 4s)
    4. BFS from perpendicular rays: spread 4s, stop at 2s/5s/8s
    """

    from collections import deque

    rows, cols = len(grid), len(grid[0])

    # Find the 3×3 block of 8s (top-left corner)
    r8, c8 = None, None
    for r in range(rows - 2):
        for c in range(cols - 2):
            if all(grid[r+dr][c+dc] == 8 for dr in range(3) for dc in range(3)):
                r8, c8 = r, c
                break
        if r8 is not None:
            break

    result = [row[:] for row in grid]

    # Shoot diagonal rays from corners (mark with 2s)
    corners = [(r8, c8, -1, -1), (r8, c8 + 2, -1, 1),
               (r8 + 2, c8, 1, -1), (r8 + 2, c8 + 2, 1, 1)]

    for cr, cc, dr, dc in corners:
        nr, nc = cr + dr, cc + dc
        while 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] in [5, 8]:
                break
            result[nr][nc] = 2
            nr += dr
            nc += dc

    # Shoot perpendicular rays from edge midpoints and collect 4-cells for BFS
    edges = [(r8 + 1, c8 - 1, 0, -1), (r8 + 1, c8 + 3, 0, 1),
             (r8 - 1, c8 + 1, -1, 0), (r8 + 3, c8 + 1, 1, 0)]

    queue = deque()
    for er, ec, dr, dc in edges:
        # Mark starting cell (adjacent to 8s block) and shoot ray
        nr, nc = er, ec
        while 0 <= nr < rows and 0 <= nc < cols:
            if grid[nr][nc] in [5, 8]:
                break
            if result[nr][nc] == 0:
                result[nr][nc] = 4
                queue.append((nr, nc))
            nr += dr
            nc += dc

    # BFS to spread 4s - stop at 2s, 5s, 8s (actual boundaries, not formulas)
    while queue:
        r, c = queue.popleft()
        # Try all 4 cardinal directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                # Stop at 2s (diagonal boundaries), 5s (markers), 8s (block)
                if result[nr][nc] == 0 and grid[nr][nc] not in [5, 8]:
                    result[nr][nc] = 4
                    queue.append((nr, nc))

    return result
