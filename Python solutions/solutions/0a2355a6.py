def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid contains 8s arranged in connected components
    2. Output replaces each component with colors 1,2,3,4
    3. Hollow rectangles (rectangles with empty interior) always get color 1
    4. Non-hollow components are assigned colors based on count:
       - If N=1 non-hollow: color 3
       - If N=2 non-hollow: colors 2, 3 (in reading order by centroid)
       - If N=3 non-hollow: colors 4, 3, 2 (in reading order)
       - Pattern: N+1, N, N-1, ..., 2 for N non-hollow components

    Procedure:
    1. Find all connected components of 8s using BFS/DFS
    2. For each component, check if it's a hollow rectangle
    3. Assign color 1 to all hollow rectangles
    4. Sort non-hollow components by centroid (row, then col)
    5. Assign colors to non-hollow based on the count
    """

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    result = [row[:] for row in grid]

    def dfs(r, c, component):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 8:
            return
        visited[r][c] = True
        component.append((r, c))
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, component)

    def is_hollow_rectangle(component):
        """Check if component forms a hollow rectangle."""
        if not component:
            return False

        # Get bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # Must be at least 3x3 to be hollow
        if height < 3 or width < 3:
            return False

        # Check expected number of cells for hollow rectangle
        expected_cells = 2 * (height + width) - 4
        if len(component) != expected_cells:
            return False

        # Verify all border cells are present and no interior cells
        component_set = set(component)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                is_border = (r == min_r or r == max_r or c == min_c or c == max_c)
                in_component = (r, c) in component_set
                if is_border != in_component:
                    return False

        return True

    # Find all components
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                component = []
                dfs(r, c, component)
                if component:
                    components.append(component)

    # Separate hollow and non-hollow components
    hollow_components = []
    non_hollow_components = []

    for comp in components:
        if is_hollow_rectangle(comp):
            hollow_components.append(comp)
        else:
            non_hollow_components.append(comp)

    # Sort non-hollow by centroid (row, then col)
    def get_centroid(comp):
        r_avg = sum(r for r, c in comp) / len(comp)
        c_avg = sum(c for r, c in comp) / len(comp)
        return (r_avg, c_avg)

    non_hollow_components.sort(key=get_centroid)

    # Assign color 1 to all hollow rectangles
    for comp in hollow_components:
        for r, c in comp:
            result[r][c] = 1

    # Assign colors to non-hollow components
    N = len(non_hollow_components)
    if N == 1:
        colors = [3]
    elif N == 2:
        colors = [2, 3]
    else:
        # N >= 3: colors N+1, N, N-1, ..., 2
        colors = list(range(N + 1, 1, -1))

    for i, comp in enumerate(non_hollow_components):
        color = colors[i] if i < len(colors) else 2
        for r, c in comp:
            result[r][c] = color

    return result
