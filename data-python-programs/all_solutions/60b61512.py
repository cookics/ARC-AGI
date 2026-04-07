def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 9x9 grid with 0s (background) and 4s (forming scattered shapes)
    2. Output has the same 4s, but 0s within bounding boxes of each component become 7s
    3. Multiple separate connected components of 4s exist in the grid
    4. Each component is processed independently with 4-connectivity

    Procedure:
    1. Find all connected components of 4s using flood fill with 4-connectivity
    2. For each component, determine its bounding rectangle (min/max row/col)
    3. Fill all 0s within each bounding rectangle with 7s
    4. Return the modified grid
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill(r, c, component):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 4:
            return
        visited[r][c] = True
        component.append((r, c))
        # Check 4 directions
        flood_fill(r + 1, c, component)
        flood_fill(r - 1, c, component)
        flood_fill(r, c + 1, component)
        flood_fill(r, c - 1, component)

    # Find all connected components of 4s
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4 and not visited[r][c]:
                component = []
                flood_fill(r, c, component)
                if component:
                    components.append(component)

    # For each component, find bounding rectangle and fill with 7s
    for component in components:
        if not component:
            continue

        # Find bounding rectangle
        min_r = min(pos[0] for pos in component)
        max_r = max(pos[0] for pos in component)
        min_c = min(pos[1] for pos in component)
        max_c = max(pos[1] for pos in component)

        # Fill all 0s within bounding rectangle with 7s
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r][c] == 0:
                    result[r][c] = 7

    return result
