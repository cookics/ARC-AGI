def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains connected components of 3s
    2. Each component gets copied exactly once
    3. Horizontal copies (left/right) use color 1
    4. Vertical copies (up/down) use color 8
    5. All cells are copied with exact relative positions

    Procedure:
    1. Find connected components
    2. For each component, try directions: right, down, left, up
    3. Use first direction where copy fits
    """
    from collections import deque

    H, W = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * W for _ in range(H)]

    def bfs(start_r, start_c):
        cells = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            cells.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc] and grid[nr][nc] == 3:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return cells

    def can_place(new_cells):
        for r, c in new_cells:
            if r < 0 or r >= H or c < 0 or c >= W or result[r][c] != 0:
                return False
        return True

    # Find all components
    components = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 3 and not visited[r][c]:
                cells = bfs(r, c)
                components.append(cells)

    def find_hollow_rect(cells):
        """Find largest hollow rectangle"""
        cell_set = set(cells)
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        best = None
        best_area = 0

        for r1 in range(min_r, max_r + 1):
            for r2 in range(r1 + 2, max_r + 1):
                for c1 in range(min_c, max_c + 1):
                    for c2 in range(c1 + 2, max_c + 1):
                        border = set()
                        is_valid = True

                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                if r == r1 or r == r2 or c == c1 or c == c2:
                                    border.add((r, c))
                                    if (r, c) not in cell_set:
                                        is_valid = False
                                        break
                                else:
                                    if (r, c) in cell_set:
                                        is_valid = False
                                        break
                            if not is_valid:
                                break

                        if is_valid:
                            area = (r2 - r1 + 1) * (c2 - c1 + 1)
                            if area > best_area:
                                best_area = area
                                best = (r1, r2, c1, c2, border)

        return best

    # Process each component
    for cells in components:
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        hollow = find_hollow_rect(cells)
        placed = False

        # Try RIGHT (color 1)
        if not placed:
            gap = 1
            if hollow:
                r1, r2, c1, c2, border = hollow
                outliers = set(cells) - border
                offset_c = c2 + gap - c1
                new_cells = []

                # Copy border
                for r, c in border:
                    new_cells.append((r, c + offset_c))

                # Reflect outliers left/right
                for r, c in outliers:
                    if c < c1:  # left → right
                        new_cells.append((r, 2*c2 + gap + (c1 - c)))
                    elif c > c2:  # right → left
                        new_cells.append((r, 2*c1 - gap - (c - c2)))
                    else:  # within range
                        new_cells.append((r, c + offset_c))
            else:
                offset_c = max_c + gap - min_c
                new_cells = [(r, c + offset_c) for r, c in cells]

            if can_place(new_cells):
                for r, c in new_cells:
                    result[r][c] = 1
                placed = True

        # Try DOWN (color 8)
        if not placed:
            gap = 1
            if hollow:
                r1, r2, c1, c2, border = hollow
                outliers = set(cells) - border
                offset_r = r2 + gap - r1
                new_cells = []

                # Copy border
                for r, c in border:
                    new_cells.append((r + offset_r, c))

                # Reflect outliers up/down
                for r, c in outliers:
                    if r < r1:  # above → below
                        new_cells.append((2*r2 + gap + (r1 - r), c))
                    elif r > r2:  # below → above
                        new_cells.append((2*r1 - gap - (r - r2), c))
                    else:  # within range
                        new_cells.append((r + offset_r, c))
            else:
                offset_r = max_r + gap - min_r
                new_cells = [(r + offset_r, c) for r, c in cells]

            if can_place(new_cells):
                for r, c in new_cells:
                    result[r][c] = 8
                placed = True

        # Try LEFT (color 1)
        if not placed:
            gap = 1
            if hollow:
                r1, r2, c1, c2, border = hollow
                outliers = set(cells) - border
                offset_c = c1 - gap - c2
                new_cells = []

                # Copy border
                for r, c in border:
                    new_cells.append((r, c + offset_c))

                # Reflect outliers
                for r, c in outliers:
                    if c > c2:  # right → left
                        new_cells.append((r, 2*c1 - gap - (c - c2)))
                    elif c < c1:  # left → right
                        new_cells.append((r, 2*c2 + gap + (c1 - c)))
                    else:
                        new_cells.append((r, c + offset_c))
            else:
                offset_c = min_c - gap - max_c
                new_cells = [(r, c + offset_c) for r, c in cells]

            if can_place(new_cells):
                for r, c in new_cells:
                    result[r][c] = 1
                placed = True

        # Try UP (color 8)
        if not placed:
            gap = 1
            if hollow:
                r1, r2, c1, c2, border = hollow
                outliers = set(cells) - border
                offset_r = r1 - gap - r2
                new_cells = []

                # Copy border
                for r, c in border:
                    new_cells.append((r + offset_r, c))

                # Reflect outliers
                for r, c in outliers:
                    if r > r2:  # below → above
                        new_cells.append((2*r1 - gap - (r - r2), c))
                    elif r < r1:  # above → below
                        new_cells.append((2*r2 + gap + (r1 - r), c))
                    else:
                        new_cells.append((r + offset_r, c))
            else:
                offset_r = min_r - gap - max_r
                new_cells = [(r + offset_r, c) for r, c in cells]

            if can_place(new_cells):
                for r, c in new_cells:
                    result[r][c] = 8
                placed = True

    return result
