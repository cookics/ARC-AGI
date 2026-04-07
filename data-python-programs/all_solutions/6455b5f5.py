def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Based on the examples, I need to find connected components of 0s that form rectangles
    and are completely isolated (surrounded by 2s or grid boundaries).

    Procedure:
    1. Find connected components of 0s using flood fill
    2. Check if each component forms a rectangle and is completely bounded
    3. Fill with 8 if area <= 6, with 1 if area > 6
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill(start_r, start_c):
        """Find all cells in the connected component of 0s"""
        component = []
        stack = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while stack:
            r, c = stack.pop()
            component.append((r, c))

            # Check all 4 directions
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr][nc]
                    and grid[nr][nc] == 0
                ):
                    visited[nr][nc] = True
                    stack.append((nr, nc))

        return component

    def is_valid_component(component):
        """Check if component is rectangular and properly bounded"""
        if not component:
            return False, None

        # Find bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        # Check if all cells in bounding box are in component (rectangular)
        expected_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        if len(component) != expected_area:
            return False, None

        # Check if completely surrounded by 2s or grid boundaries
        component_set = set(component)

        for r, c in component:
            # Check all 4 directions
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                # If outside grid, that's okay (grid boundary)
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue

                # If adjacent cell is not in component, it must be a 2
                if (nr, nc) not in component_set:
                    if grid[nr][nc] != 2:
                        return False, None

        return True, (min_r, min_c, max_r, max_c)

    # Find all components and fill valid ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and not visited[r][c]:
                component = flood_fill(r, c)
                is_valid, bounds = is_valid_component(component)

                if is_valid:
                    min_r, min_c, max_r, max_c = bounds
                    area = len(component)
                    fill_value = 8 if area <= 6 else 1

                    for r_fill in range(min_r, max_r + 1):
                        for c_fill in range(min_c, max_c + 1):
                            result[r_fill][c_fill] = fill_value

    return result
