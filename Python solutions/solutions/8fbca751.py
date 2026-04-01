def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing values 0 and 8
    2. Output is a 2D grid with values 0, 2, and 8
    3. The 8s form one or more connected components (4-connectivity)
    4. For each connected component of 8s, we find its bounding box
    5. All 0s within each bounding box are replaced with 2s
    6. Isolated 8s (single cells) are merged with nearest larger component

    Procedure:
    1. Use flood fill to find all connected components of 8s
    2. Separate isolated components (size 1) from larger components
    3. Merge each isolated component with its nearest larger component
    4. For each final component, calculate bounding box (min/max row and column)
    5. Within each bounding box, replace all 0s with 2s while keeping 8s unchanged
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill(r, c, component):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 8:
            return
        visited[r][c] = True
        component.append((r, c))
        # 4-connectivity
        flood_fill(r - 1, c, component)
        flood_fill(r + 1, c, component)
        flood_fill(r, c - 1, component)
        flood_fill(r, c + 1, component)

    components = []

    # Find all connected components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                component = []
                flood_fill(r, c, component)
                if component:
                    components.append(component)

    # Merge isolated components (size 1) with nearest larger component
    isolated_components = [comp for comp in components if len(comp) == 1]
    large_components = [comp for comp in components if len(comp) > 1]

    for isolated in isolated_components:
        if not large_components:
            large_components.append(isolated)
            continue

        # Find nearest large component
        isolated_r, isolated_c = isolated[0]
        min_dist = float("inf")
        nearest_comp = None

        for large_comp in large_components:
            for comp_r, comp_c in large_comp:
                dist = abs(isolated_r - comp_r) + abs(
                    isolated_c - comp_c
                )  # Manhattan distance
                if dist < min_dist:
                    min_dist = dist
                    nearest_comp = large_comp

        # Merge isolated component with nearest large component
        if nearest_comp is not None:
            nearest_comp.extend(isolated)

    # Use only the large components (which now include merged isolated ones)
    final_components = large_components

    # For each component, fill its bounding box
    for component in final_components:
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        # Fill the bounding box with 2s (replace 0s only)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r][c] == 0:
                    result[r][c] = 2

    return result
