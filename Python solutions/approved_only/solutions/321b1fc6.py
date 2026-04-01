def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing 0s, 8s, and other colored values.
    2. There exists exactly one template pattern made of non-8, non-0 values.
    3. Multiple connected components of 8s are scattered throughout the grid.
    4. Output replaces each connected component of 8s with the template pattern positioned at the top-left corner of where the 8s were.

    Procedure:
    1. Identify and extract the template pattern by finding all non-8, non-0 values and their bounding box.
    2. Find all connected components of 8s using flood fill algorithm.
    3. For each 8-component, determine its top-left corner position.
    4. Place the template pattern at each 8-component's top-left position in the output grid.
    """

    rows, cols = len(grid), len(grid[0])

    # Find template pattern
    template_cells = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and grid[i][j] != 8:
                template_cells.append((i, j))

    if not template_cells:
        return grid

    # Find bounding box of template
    min_row = min(r for r, c in template_cells)
    max_row = max(r for r, c in template_cells)
    min_col = min(c for r, c in template_cells)
    max_col = max(c for r, c in template_cells)

    # Extract template pattern
    template_height = max_row - min_row + 1
    template_width = max_col - min_col + 1
    template = []
    for i in range(template_height):
        row = []
        for j in range(template_width):
            val = grid[min_row + i][min_col + j]
            if val != 0 and val != 8:
                row.append(val)
            else:
                row.append(0)
        template.append(row)

    # Create output grid
    output = [[0] * cols for _ in range(rows)]

    # Find connected components of 8s using flood fill
    visited = [[False] * cols for _ in range(rows)]

    def get_component(start_r, start_c):
        component = []
        stack = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while stack:
            r, c = stack.pop()
            component.append((r, c))

            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr][nc]
                    and grid[nr][nc] == 8
                ):
                    visited[nr][nc] = True
                    stack.append((nr, nc))

        return component

    # Find all 8-components and replace with template
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 8 and not visited[i][j]:
                component = get_component(i, j)

                # Find top-left corner of component
                min_r = min(r for r, c in component)
                min_c = min(c for r, c in component)

                # Place template at this position
                for ti in range(template_height):
                    for tj in range(template_width):
                        if template[ti][tj] != 0:
                            target_r = min_r + ti
                            target_c = min_c + tj
                            if 0 <= target_r < rows and 0 <= target_c < cols:
                                output[target_r][target_c] = template[ti][tj]

    return output
