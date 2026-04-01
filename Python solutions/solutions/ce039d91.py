def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 0s and 5s
    2. Find connected components of 5s using BFS
    3. For each component, find the largest filled rectangle
    4. Cells in that rectangle become 1, cells outside stay as 5

    Procedure:
    1. Use BFS to find connected components
    2. For each component with size >= 2, find largest filled rectangle
    3. Mark rectangle cells as 1, non-rectangle cells as 5
    4. For single-cell components, mark as 1
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    def bfs(start_r, start_c):
        """Find connected component using BFS"""
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] == 5 and not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return component

    def find_maximal_rectangles(component):
        """Find all maximal rectangles - those that can't be extended"""
        if len(component) < 2:
            return set()

        comp_set = set(component)
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        all_rects = []

        # Find all filled rectangles
        for r1 in range(min_r, max_r + 1):
            for c1 in range(min_c, max_c + 1):
                if (r1, c1) not in comp_set:
                    continue

                for r2 in range(r1, max_r + 1):
                    for c2 in range(c1, max_c + 1):
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area < 2:
                            continue

                        # Check if filled
                        is_filled = True
                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                if (r, c) not in comp_set:
                                    is_filled = False
                                    break
                            if not is_filled:
                                break

                        if is_filled:
                            all_rects.append((r1, c1, r2, c2))

        # Filter to maximal rectangles only (not contained in any other)
        maximal_rects = []
        for i, (r1a, c1a, r2a, c2a) in enumerate(all_rects):
            is_maximal = True
            for j, (r1b, c1b, r2b, c2b) in enumerate(all_rects):
                if i != j:
                    # Check if rect A is strictly contained in rect B
                    if r1b <= r1a and r2a <= r2b and c1b <= c1a and c2a <= c2b:
                        if (r1b, c1b, r2b, c2b) != (r1a, c1a, r2a, c2a):
                            is_maximal = False
                            break
            if is_maximal:
                maximal_rects.append((r1a, c1a, r2a, c2a))

        # Collect all cells in maximal rectangles
        rect_cells = set()
        for r1, c1, r2, c2 in maximal_rects:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    rect_cells.add((r, c))

        return rect_cells

    # Process each connected component
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                component = bfs(r, c)
                rect_cells = find_largest_rectangle(component)

                for cr, cc in component:
                    if (cr, cc) in rect_cells:
                        result[cr][cc] = 1
                    else:
                        result[cr][cc] = 5

    return result
