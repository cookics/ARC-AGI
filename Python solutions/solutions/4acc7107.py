def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with colored connected components
    2. For each color, there are typically 2 components that swap vertical positions
    3. Components preserve their exact shapes during transformation
    4. Top components of each color move to bottom, bottom components move to top
    5. Horizontal positions are preserved relative to left/right side of grid
    6. Components on the middle row (row 5) tend to stay in place

    Procedure:
    1. Find all connected components using BFS
    2. Group components by color
    3. For each color group, pair top and bottom components
    4. Swap their vertical positions while preserving shapes
    5. Adjust horizontal alignment to keep left components left, right components right
    """

    n = len(grid)
    m = len(grid[0])

    # Find connected components
    visited = [[False] * m for _ in range(n)]
    components = []

    def bfs(start_r, start_c, color):
        queue = [(start_r, start_c)]
        component = []
        visited[start_r][start_c] = True
        while queue:
            r, c = queue.pop(0)
            component.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < m and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component

    for r in range(n):
        for c in range(m):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                component = bfs(r, c, color)
                components.append((color, component))

    result = [[0] * m for _ in range(n)]

    # Group components by color
    color_groups = {}
    for color, component in components:
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(component)

    # Process each color group
    for color, comps in color_groups.items():
        # Sort components by their average row position
        comp_info = []
        for comp in comps:
            rows = [r for r, c in comp]
            cols = [c for r, c in comp]
            avg_row = sum(rows) / len(rows)
            min_row, max_row = min(rows), max(rows)
            min_col = min(cols)
            comp_info.append((avg_row, min_row, max_row, min_col, comp))

        comp_info.sort()  # Sort by avg_row

        # Pair and swap: first (top) with last (bottom), etc.
        for i in range(len(comp_info)):
            avg_row, min_row, max_row, min_col, comp = comp_info[i]

            # Determine if this is in top half or bottom half
            if max_row < 5:  # Top half
                # Move down so max_row becomes 9
                row_shift = 9 - max_row
            elif min_row > 5:  # Bottom half
                # Move up so max_row becomes 5
                row_shift = 5 - max_row
            elif min_row == 5 and max_row == 5:  # Exactly row 5
                row_shift = 0
            else:
                # Straddles middle - decide based on avg_row
                if avg_row < 5:
                    row_shift = 9 - max_row  # Treat as top
                else:
                    row_shift = 5 - max_row  # Treat as bottom

            # Calculate new position
            new_min_row = min_row + row_shift
            new_max_row = max_row + row_shift

            # Adjust if out of bounds
            if new_min_row < 0:
                adjustment = -new_min_row
                row_shift += adjustment
            elif new_max_row >= n:
                adjustment = new_max_row - (n - 1)
                row_shift -= adjustment

            # Determine horizontal shift based on original position and color
            if row_shift < 0:  # Moving up
                # Components moving up align to left (col 0)
                col_shift = -min_col
            elif row_shift > 0:  # Moving down
                # Components moving down: left stays left, right goes to ~col 5
                if min_col < 3:
                    col_shift = -min_col  # Keep on left (align to col 0)
                elif min_col >= 6:
                    # Far right: align to col 5 or 6
                    col_shift = 6 - min_col
                else:
                    # Middle-right: align to col 5
                    col_shift = 5 - min_col
            else:
                col_shift = 0

            # Place component
            for r, c in comp:
                new_r = r + row_shift
                new_c = c + col_shift
                if 0 <= new_r < n and 0 <= new_c < m:
                    result[new_r][new_c] = color

    return result
