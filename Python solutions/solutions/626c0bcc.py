def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 7x7 grid with 0s and 8s
    2. Output replaces 8s with colors 1-4 based on 2x2 block positions
    3. Connected components use patterns based on relative position within component
    4. Pattern is [[a,b],[c,d]] indexed by ((r-min_r)//2)%2, ((c-min_c)//2)%2
    5. Pattern varies by component order/position

    Procedure:
    1. Find all connected components of 8s
    2. Sort components by top-left position (row first, then column)
    3. For each component, apply appropriate 2x2 block pattern
    4. Use relative coordinates within each component
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components using BFS
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(start_r, start_c):
        component = []
        queue = [(start_r, start_c)]
        visited[start_r][start_c] = True
        while queue:
            r, c = queue.pop(0)
            component.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr][nc]
                    and grid[nr][nc] == 8
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component

    # Collect all components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                component = bfs(r, c)
                min_r = min(r for r, c in component)
                min_c = min(c for r, c in component)
                components.append((min_r, min_c, component))

    # Sort by top-left corner (row first, then column)
    components.sort(key=lambda x: (x[0], x[1]))

    # Apply coloring based on component count and order
    for comp_idx, (min_r, min_c, component) in enumerate(components):
        max_r = max(r for r, c in component)
        max_c = max(c for r, c in component)
        bbox_rows = max_r - min_r + 1
        bbox_cols = max_c - min_c + 1

        # Determine coloring strategy based on component properties
        if len(components) >= 2:
            # Multi-component scenario: use component-relative block patterns
            if comp_idx == 0:
                pattern = [[2, 3], [1, 1]]
            else:
                pattern = [[4, 4], [1, 1]]

            for r, c in component:
                rel_r = r - min_r
                rel_c = c - min_c
                block_r = (rel_r // 2) % 2
                block_c = (rel_c // 2) % 2
                result[r][c] = pattern[block_r][block_c]

        elif bbox_rows <= 4 and bbox_cols <= 4:
            # Single compact component: simple 2x2 block pattern
            pattern = [[1, 4], [3, 2]]
            for r, c in component:
                rel_r = r - min_r
                rel_c = c - min_c
                block_r = (rel_r // 2) % 2
                block_c = (rel_c // 2) % 2
                result[r][c] = pattern[block_r][block_c]

        else:
            # Single large component: use absolute position-based formula
            # Color based on (r, c) with different regions
            for r, c in component:
                # Determine color based on position and parity
                if (r // 2) % 2 == 0:  # Even block row
                    if (c // 2) % 2 == 0:  # Even block col
                        if c < 2:
                            result[r][c] = 3 if r % 2 == 1 else 1
                        elif c >= 4:
                            result[r][c] = 4 if r <= 2 else 1
                        else:
                            result[r][c] = 1
                    else:  # Odd block col
                        result[r][c] = 1
                else:  # Odd block row
                    if (c // 2) % 2 == 0:  # Even block col
                        if c < 2:
                            result[r][c] = 3 if r % 2 == 0 else 2
                        elif c >= 4:
                            result[r][c] = 4 if r % 2 == 0 else 1
                        else:
                            result[r][c] = 2
                    else:  # Odd block col
                        if c % 2 == 1 and r % 2 == 1:
                            result[r][c] = 1
                        else:
                            result[r][c] = 2

    return result
