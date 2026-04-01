def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with non-zero colored regions and zero background
    2. Output is the same grid where minority colors within each connected component are replaced with 0
    3. Connected components use 8-connectivity (including diagonals)
    4. Within each component, the most frequent color is dominant and stays, others become 0

    Procedure:
    1. Find all connected components of non-zero pixels using DFS with 8-connectivity
    2. For each component, count the frequency of each color
    3. Keep only the most frequent color, replace all others with 0
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy
    visited = [[False] * cols for _ in range(rows)]

    def get_neighbors(r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append((nr, nc))
        return neighbors

    def dfs(r, c, component):
        if visited[r][c] or grid[r][c] == 0:
            return
        visited[r][c] = True
        component.append((r, c))

        for nr, nc in get_neighbors(r, c):
            if not visited[nr][nc] and grid[nr][nc] != 0:
                dfs(nr, nc, component)

    # Find all connected components
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                component = []
                dfs(r, c, component)

                # Count colors in this component
                color_count = {}
                for cr, cc in component:
                    color = grid[cr][cc]
                    color_count[color] = color_count.get(color, 0) + 1

                # Find dominant color
                dominant_color = max(color_count, key=color_count.get)

                # Replace non-dominant colors with 0
                for cr, cc in component:
                    if grid[cr][cc] != dominant_color:
                        result[cr][cc] = 0

    return result
