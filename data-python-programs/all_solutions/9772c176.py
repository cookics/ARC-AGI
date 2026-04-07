def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains diamond/octagonal shapes made of 8s on a background of 0s
    2. Output preserves all 8s and adds decorative 4s around each shape:
       - Horizontal bulges (4s) on left/right sides at the vertical center
       - Triangular extensions (4s) above and below that narrow as they extend away
    3. Each shape is a separate connected component of 8s
    4. Bulge size relates to shape height: max_bulge = height // 4
    5. Triangular width decreases by 2 for each row away from shape edge

    Procedure:
    1. Find all connected components of 8s using BFS
    2. For each component:
       a. Find bounding box (min/max row/col) and vertical center
       b. Add horizontal bulges on sides (size decreases with distance from center)
       c. Add triangular extensions above top edge (narrowing upward)
       d. Add triangular extensions below bottom edge (narrowing downward)
    """
    from collections import deque

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Copy grid

    # Find connected components of 8s using BFS
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        component = [(start_r, start_c)]

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 8:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    component.append((nr, nc))

        return component

    # Find all components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                component = bfs(r, c)
                components.append(component)

    # Process each component
    for component in components:
        # Find bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)

        height = max_r - min_r + 1

        # Find center columns for triangular extensions
        top_row_cols = [c for r, c in component if r == min_r]
        bottom_row_cols = [c for r, c in component if r == max_r]
        top_center_c = (min(top_row_cols) + max(top_row_cols)) // 2
        bottom_center_c = (min(bottom_row_cols) + max(bottom_row_cols)) // 2

        # Add side bulges
        max_bulge = height // 4
        if max_bulge > 0:
            # Find widest rows (rows with most cells in component)
            row_widths = {}
            for r in range(min_r, max_r + 1):
                row_cols = [c for r2, c in component if r2 == r]
                if row_cols:
                    row_widths[r] = len(row_cols)

            max_width = max(row_widths.values())
            widest_rows = [r for r, w in row_widths.items() if w == max_width]
            widest_center_r = (min(widest_rows) + max(widest_rows)) / 2

            for r in range(min_r, max_r + 1):
                # Bulge size decreases with distance from center of widest rows
                bulge_size = max_bulge - int(abs(r - widest_center_r))
                if bulge_size > 0:
                    # Find leftmost and rightmost 8 in this row
                    row_cols = [c for r2, c in component if r2 == r]
                    if row_cols:
                        left_c = min(row_cols)
                        right_c = max(row_cols)

                        # Add bulges extending outward
                        for i in range(1, bulge_size + 1):
                            if left_c - i >= 0:
                                result[r][left_c - i] = 4
                            if right_c + i < cols:
                                result[r][right_c + i] = 4

        # Add top triangular extension (narrowing upward)
        top_width = max(top_row_cols) - min(top_row_cols) + 1
        dist = 1
        while top_width - 2 * dist > 0:
            width = top_width - 2 * dist
            target_r = min_r - dist
            if target_r >= 0:
                start_c = top_center_c - width // 2
                for i in range(width):
                    if 0 <= start_c + i < cols:
                        result[target_r][start_c + i] = 4
            dist += 1

        # Add bottom triangular extension (narrowing downward)
        bottom_width = max(bottom_row_cols) - min(bottom_row_cols) + 1
        dist = 1
        while bottom_width - 2 * dist > 0:
            width = bottom_width - 2 * dist
            target_r = max_r + dist
            if target_r < rows:
                start_c = bottom_center_c - width // 2
                for i in range(width):
                    if 0 <= start_c + i < cols:
                        result[target_r][start_c + i] = 4
            dist += 1

    return result
