def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with rectangular regions filled with non-zero values (1, 2, 8)
    2. These regions are separated by 0s (background)
    3. Output is one specific rectangular region that is "isolated" (surrounded by 0s or grid boundaries)
    4. Among all isolated rectangular regions, select the one whose center is farthest from the grid center

    Procedure:
    1. Find all connected components of non-zero values using flood fill
    2. For each component, check if it forms a filled rectangle
    3. For each filled rectangle, check if it's isolated (surrounded by 0s or boundaries)
    4. Among isolated regions, find the one with center farthest from grid center
    5. Extract and return that region
    """

    rows = len(grid)
    cols = len(grid[0])
    grid_center_r = (rows - 1) / 2
    grid_center_c = (cols - 1) / 2

    visited = [[False] * cols for _ in range(rows)]
    regions = []

    def flood_fill(start_r, start_c):
        """Find all connected non-zero cells from starting position"""
        cells = []
        stack = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while stack:
            r, c = stack.pop()
            cells.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr][nc] and grid[nr][nc] != 0:
                        visited[nr][nc] = True
                        stack.append((nr, nc))

        return cells

    # Find all connected components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = flood_fill(r, c)

                # Find bounding box
                min_r = min(cell[0] for cell in cells)
                max_r = max(cell[0] for cell in cells)
                min_c = min(cell[1] for cell in cells)
                max_c = max(cell[1] for cell in cells)

                # Check if it's a filled rectangle (all cells in bbox are part of component)
                expected_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
                if len(cells) == expected_cells:
                    # It's a filled rectangle, now check if it's isolated
                    isolated = True

                    # Check top and bottom edges
                    for col in range(min_c, max_c + 1):
                        if min_r > 0 and grid[min_r - 1][col] != 0:
                            isolated = False
                            break
                        if max_r < rows - 1 and grid[max_r + 1][col] != 0:
                            isolated = False
                            break

                    # Check left and right edges
                    if isolated:
                        for row in range(min_r, max_r + 1):
                            if min_c > 0 and grid[row][min_c - 1] != 0:
                                isolated = False
                                break
                            if max_c < cols - 1 and grid[row][max_c + 1] != 0:
                                isolated = False
                                break

                    if isolated:
                        regions.append((min_r, max_r, min_c, max_c))

    # Find region farthest from grid center
    max_distance = -1
    best_region = None

    for min_r, max_r, min_c, max_c in regions:
        center_r = (min_r + max_r) / 2
        center_c = (min_c + max_c) / 2
        distance = ((center_r - grid_center_r) ** 2 + (center_c - grid_center_c) ** 2) ** 0.5

        if distance > max_distance:
            max_distance = distance
            best_region = (min_r, max_r, min_c, max_c)

    # Extract the region
    min_r, max_r, min_c, max_c = best_region
    result = []
    for r in range(min_r, max_r + 1):
        result.append(grid[r][min_c:max_c + 1])

    return result
