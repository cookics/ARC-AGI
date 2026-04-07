def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 20x20 grid with a main structured region and scattered cells
    2. Output is a rotated version of the extracted main region
    3. Transformation: 90-degree counter-clockwise rotation
    4. Formula: (r, c) → (width-1-c, r)

    Procedure:
    1. Find all 8-connected components
    2. Identify the largest component (main pattern)
    3. Extract the bounding box of the main pattern
    4. Apply 90-degree counter-clockwise rotation
    """

    rows, cols = len(grid), len(grid[0])

    # Find all 8-connected components
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, component):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] == 0:
            return
        visited[r][c] = True
        component.append((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr != 0 or dc != 0:
                    dfs(r + dr, c + dc, component)

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid[i][j] != 0:
                component = []
                dfs(i, j, component)
                if component:
                    components.append(component)

    if not components:
        return []

    # Find the largest component
    largest_component = max(components, key=len)

    # Find bounding box
    min_row = min(pos[0] for pos in largest_component)
    max_row = max(pos[0] for pos in largest_component)
    min_col = min(pos[1] for pos in largest_component)
    max_col = max(pos[1] for pos in largest_component)

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    # Extract the region
    extracted = []
    for r in range(min_row, max_row + 1):
        row = []
        for c in range(min_col, max_col + 1):
            row.append(grid[r][c])
        extracted.append(row)

    # Apply 90-degree counter-clockwise rotation
    # (r, c) → (width-1-c, r)
    result = [[0] * height for _ in range(width)]
    for r in range(height):
        for c in range(width):
            new_r = width - 1 - c
            new_c = r
            result[new_r][new_c] = extracted[r][c]

    return result
