def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. 0-bordered rectangles contain 5s and/or 7s in their interiors
    2. 9-markers indicate reference points for transformations
    3. Rectangles near 9-markers may swap positions or move
    4. When moving, interiors transform: 5s→7s or 7s expand
    5. Movement direction (vertical/horizontal) affects transformation type

    Procedure:
    1. Identify all 0-bordered rectangles and extract their properties
    2. For each rectangle, check for nearby 9-markers
    3. Apply movement and transformation rules
    4. Update grid with transformed rectangles
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    def find_rects():
        visited = [[False] * cols for _ in range(rows)]
        rects = []

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0 and not visited[r][c]:
                    q = deque([(r, c)])
                    visited[r][c] = True
                    cells = [(r, c)]

                    while q:
                        cr, cc = q.popleft()
                        for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < rows and 0 <= nc < cols and
                                not visited[nr][nc] and grid[nr][nc] == 0):
                                visited[nr][nc] = True
                                q.append((nr, nc))
                                cells.append((nr, nc))

                    if len(cells) >= 9:
                        rs = [r for r, c in cells]
                        cs = [c for r, c in cells]
                        r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)

                        if r2 - r1 >= 2 and c2 - c1 >= 2:
                            has_interior = any(grid[ir][ic] in [5, 7]
                                             for ir in range(r1+1, r2)
                                             for ic in range(c1+1, c2))
                            if has_interior:
                                interior = [[grid[ir][ic] for ic in range(c1+1, c2)]
                                          for ir in range(r1+1, r2)]
                                rects.append((r1, r2, c1, c2, interior))

        return rects

    rects = find_rects()

    # For each rectangle, apply transformations
    for r1, r2, c1, c2, interior in rects:
        h, w = len(interior), len(interior[0])
        has_7 = any(interior[r][c] == 7 for r in range(h) for c in range(w))

        # Check for 9-markers immediately adjacent (within 2 cells)
        has_9_above = any(grid[r][c] == 9 for r in range(max(0, r1-2), r1)
                         for c in range(c1, c2+1))
        has_9_below = any(grid[r][c] == 9 for r in range(r2+1, min(rows, r2+3))
                         for c in range(c1, c2+1))
        has_9_left = any(grid[r][c] == 9 for r in range(r1, r2+1)
                        for c in range(max(0, c1-2), c1))
        has_9_right = any(grid[r][c] == 9 for r in range(r1, r2+1)
                         for c in range(c2+1, min(cols, c2+3)))

        moved = False
        new_r1, new_r2, new_c1, new_c2 = r1, r2, c1, c2
        new_interior = [row[:] for row in interior]

        # Prioritize horizontal movement
        if has_9_right:
            # Move right across the 9
            # Find first 9 column
            nine_col = None
            for c in range(c2+1, min(cols, c2+4)):
                if any(grid[r][c] == 9 for r in range(r1, r2+1)):
                    nine_col = c
                    break

            if nine_col:
                # Move to where the 9 was
                new_c1, new_c2 = nine_col, nine_col + (c2 - c1)
                moved = True
                # Transform based on content
                if not has_7:
                    new_interior = [[7] * w for _ in range(h)]
                    if h > 1:
                        new_interior[-1][0] = 5
                else:
                    # Transform: top row all 7s, other rows keep first 5 and fill rest with 7
                    new_interior = [[7] * w for _ in range(h)]
                    for r in range(1, h):
                        if interior[r][0] == 5:
                            new_interior[r][0] = 5

        elif has_9_left:
            # Move left
            nine_col = None
            for c in range(max(0, c1-3), c1):
                if any(grid[r][c] == 9 for r in range(r1, r2+1)):
                    nine_col = c

            if nine_col is not None:
                new_c2 = nine_col - 1
                new_c1 = new_c2 - (c2 - c1)
                moved = True
                # Transform
                if not has_7:
                    new_interior = [[7] * w for _ in range(h)]
                    if h > 1:
                        new_interior[-1][0] = 5
                else:
                    # Expand 7s to the right (opposite of move direction)
                    temp = [row[:] for row in interior]
                    for r in range(h):
                        for c in range(w):
                            if interior[r][c] == 7:
                                # Expand right
                                for dc in range(1, w - c):
                                    if c + dc < w and interior[r][c+dc] == 5:
                                        temp[r][c+dc] = 7
                                    else:
                                        break
                    new_interior = temp

        elif has_9_above:
            # Move up by 1
            new_r1, new_r2 = r1 - 1, r2 - 1
            moved = True
            # Transform: if has 7, expand; else add 7 at top-left
            if has_7:
                for r in range(h):
                    for c in range(w):
                        if interior[r][c] == 7:
                            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                                nr, nc = r+dr, c+dc
                                if 0 <= nr < h and 0 <= nc < w and interior[nr][nc] == 5:
                                    new_interior[nr][nc] = 7
            else:
                new_interior[0][0] = 7

        if moved and 0 <= new_r1 < rows and 0 <= new_r2 < rows and 0 <= new_c1 < cols and 0 <= new_c2 < cols:
            # Clear old position
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    result[r][c] = 6

            # Draw new rectangle
            for c in range(new_c1, new_c2+1):
                if 0 <= new_r1 < rows:
                    result[new_r1][c] = 0
                if 0 <= new_r2 < rows:
                    result[new_r2][c] = 0
            for r in range(new_r1, new_r2+1):
                if 0 <= new_c1 < cols:
                    result[r][new_c1] = 0
                if 0 <= new_c2 < cols:
                    result[r][new_c2] = 0

            # Fill interior
            for i in range(min(h, new_r2 - new_r1)):
                for j in range(min(w, new_c2 - new_c1)):
                    nr, nc = new_r1 + 1 + i, new_c1 + 1 + j
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = new_interior[i][j]

    return result
