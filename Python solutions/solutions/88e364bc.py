def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with hollow shapes (regions bounded by colored values)
    2. Each 4 is inside a hollow shape and needs to move to the inner wall
    3. The 4 moves to a position adjacent to the colored boundary, staying on the same row when possible
    4. When multiple candidates exist, prefer positions farther from grid edges

    Procedure:
    1. Identify "inside" regions using flood fill from edges
    2. For each 4, search on same row first for positions adjacent to boundaries
    3. If none on same row, search nearby rows
    4. Choose position based on: inside region, adjacent to boundary, farther from edges
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all 4s
    fours = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                fours.append((r, c))
                result[r][c] = 0

    # Flood fill from edges to identify "outside" region
    outside = [[False] * cols for _ in range(rows)]
    queue = deque()

    for r in range(rows):
        for c in range(cols):
            if (r == 0 or r == rows - 1 or c == 0 or c == cols - 1):
                if grid[r][c] in [0, 4]:
                    queue.append((r, c))
                    outside[r][c] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not outside[nr][nc] and grid[nr][nc] in [0, 4]:
                    outside[nr][nc] = True
                    queue.append((nr, nc))

    # For each 4, find best position
    for (r4, c4) in fours:
        if outside[r4][c4]:
            result[r4][c4] = 4
            continue

        # BFS to find candidates
        queue2 = deque([(r4, c4, 0)])
        visited = {(r4, c4)}
        candidates = []

        while queue2:
            cr, cc, dist = queue2.popleft()

            if dist > 10:
                break

            # Check if adjacent to boundary
            is_adjacent = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] not in [0, 4]:
                        is_adjacent = True
                        break

            if is_adjacent:
                same_row = (cr == r4)
                # Prefer positions farther from original (negative distance means prefer larger dist)
                candidates.append((same_row, dist, cr, cc))

            # Continue BFS
            if dist < 10:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if (nr, nc) not in visited and not outside[nr][nc] and grid[nr][nc] in [0, 4]:
                            visited.add((nr, nc))
                            queue2.append((nr, nc, dist + 1))

        # Choose best: prefer same row, then farther distance
        if candidates:
            candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
            result[candidates[0][2]][candidates[0][3]] = 4
        else:
            result[r4][c4] = 4

    return result
