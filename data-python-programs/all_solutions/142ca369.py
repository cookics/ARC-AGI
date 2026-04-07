def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has colored connected components
    2. Each component extends diagonally based on its type:
       - Vertical lines: spread horizontally then extend with two diagonals
       - Horizontal lines: extend with two diagonals
       - L-shapes: extend with one diagonal from corner
       - Multiple components: two diagonals connecting them

    Procedure:
    1. Find connected components by color
    2. Classify each component (vertical, horizontal, or other)
    3. Draw appropriate diagonal extensions
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    def bfs(start_r, start_c, color, visited):
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return component

    visited = [[False] * cols for _ in range(rows)]
    components_by_color = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                comp = bfs(r, c, color, visited)
                if color not in components_by_color:
                    components_by_color[color] = []
                components_by_color[color].append(comp)

    def draw_diagonal(start_r, start_c, dr, dc, color):
        r, c = start_r, start_c
        while 0 <= r < rows and 0 <= c < cols:
            if result[r][c] == 0:
                result[r][c] = color
            r += dr
            c += dc

    def get_bounds(comp):
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        return min(rs), max(rs), min(cs), max(cs)

    def is_vertical(comp):
        min_r, max_r, min_c, max_c = get_bounds(comp)
        return min_c == max_c and max_r - min_r >= 2

    def is_horizontal(comp):
        min_r, max_r, min_c, max_c = get_bounds(comp)
        return min_r == max_r and max_c - min_c >= 2

    # Process each color
    for color, components in components_by_color.items():
        if len(components) == 1:
            comp = components[0]
            min_r, max_r, min_c, max_c = get_bounds(comp)
            mid_r = (min_r + max_r) // 2
            mid_c = (min_c + max_c) // 2

            if is_vertical(comp):
                # Vertical line: create spread then diagonals
                if min_c < cols // 2:
                    dc_spread = 1
                else:
                    dc_spread = -1

                for r, c in comp:
                    dist_from_mid = abs(r - mid_r)
                    spread = 2 if dist_from_mid > 0 else 1
                    nc = c + spread * dc_spread
                    if 0 <= nc < cols and result[r][nc] == 0:
                        result[r][nc] = color

                # Draw two diagonals
                if dc_spread > 0:
                    draw_diagonal(mid_r, min_c + 1, -1, -1, color)
                    draw_diagonal(max_r + 1, min_c + 2, 1, 1, color)
                else:
                    draw_diagonal(mid_r, min_c - 1, -1, 1, color)
                    draw_diagonal(max_r + 1, min_c - 2, 1, -1, color)

            elif is_horizontal(comp):
                # Horizontal line: two diagonals from center
                draw_diagonal(min_r + 1, mid_c, 1, 1, color)
                draw_diagonal(min_r - 1, mid_c + 1, -1, 1, color)

            else:
                # L-shape or cluster: single diagonal from corner
                draw_diagonal(max_r + 1, max_c + 1, 1, 1, color)

        else:
            # Multiple components: two diagonals connecting them
            components.sort(key=lambda c: (min(r for r,_ in c), min(c for _,c in c)))
            comp1 = components[0]
            comp2 = components[1]

            min_r1, max_r1, min_c1, max_c1 = get_bounds(comp1)
            min_r2, max_r2, min_c2, max_c2 = get_bounds(comp2)

            # Draw slope +1 diagonal from first component
            draw_diagonal(max_r1 + 1, max_c1 + 1, 1, 1, color)

            # Draw slope -1 diagonal intersecting at point between components
            ref_r = min_r2 - 1
            ref_c = min_c2
            draw_diagonal(ref_r + 1, ref_c - 1, 1, -1, color)

    return result
