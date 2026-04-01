def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing 0s and 8s
    2. Output is a grid where 8s are replaced with 1s or 2s based on connected component size
    3. Connected components are formed by 4-connectivity (orthogonal neighbors only)
    4. The smallest connected component(s) get value 2, all larger components get value 1

    Procedure:
    1. Find all connected components of 8s using flood fill (DFS or BFS)
    2. Calculate the size of each component
    3. Find the minimum size among all components
    4. Replace each component: if size equals minimum, use value 2, otherwise use value 1
    """

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    result = [[0] * cols for _ in range(rows)]

    def flood_fill(start_r, start_c):
        """Returns list of (r,c) coordinates in the connected component"""
        if visited[start_r][start_c] or grid[start_r][start_c] != 8:
            return []

        component = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != 8:
                continue

            visited[r][c] = True
            component.append((r, c))

            # Add 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))

        return component

    # Find all connected components
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                component = flood_fill(r, c)
                if component:
                    components.append(component)

    # Find the component with minimum size
    if not components:
        return result

    min_size = min(len(comp) for comp in components)

    # Transform components
    for component in components:
        value = 2 if len(component) == min_size else 1
        for r, c in component:
            result[r][c] = value

    return result
