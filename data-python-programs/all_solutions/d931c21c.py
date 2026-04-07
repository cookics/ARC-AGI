def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains connected components of 1s forming hollow shapes
    2. Output transforms closed hollow shapes by:
       - Adding a border of 2s around the bounding box
       - Filling interior cells (enclosed by 1s) with 3s
       - Keeping interior cells with distance >= 2 from nearest 1 as 0
    3. Shapes with gaps (interior connects to exterior) remain unchanged

    Procedure:
    1. Find connected components of 1s using BFS
    2. For each component:
       a. Use flood fill from grid borders to identify exterior cells
       b. Find interior cells (not reachable from outside)
       c. If there are interior cells:
          - Add border of 2s (exterior cells adjacent to component/interior)
          - Fill interior cells based on distance to nearest 1
    """
    rows = len(grid)
    cols = len(grid[0])

    # Find connected components of 1s
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs_component(start_r, start_c):
        queue = [(start_r, start_c)]
        component = []
        visited[start_r][start_c] = True
        while queue:
            r, c = queue.pop(0)
            component.append((r, c))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 1:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return component

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                comp = bfs_component(r, c)
                components.append(comp)

    # Create result grid
    result = [row[:] for row in grid]

    # Process each component
    for comp in components:
        # Find bounding box
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        # Flood fill from outside to find exterior cells
        exterior = [[False] * cols for _ in range(rows)]
        visited_flood = [[False] * cols for _ in range(rows)]

        # Start from all border cells that are 0
        queue = []
        for r in range(rows):
            for c in [0, cols-1]:
                if grid[r][c] == 0 and not visited_flood[r][c]:
                    queue.append((r, c))
                    visited_flood[r][c] = True
                    exterior[r][c] = True
        for c in range(cols):
            for r in [0, rows-1]:
                if grid[r][c] == 0 and not visited_flood[r][c]:
                    queue.append((r, c))
                    visited_flood[r][c] = True
                    exterior[r][c] = True

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited_flood[nr][nc] and grid[nr][nc] == 0:
                    visited_flood[nr][nc] = True
                    exterior[nr][nc] = True
                    queue.append((nr, nc))

        # Find interior cells within bounding box
        interior_cells = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r][c] == 0 and not exterior[r][c]:
                    interior_cells.append((r, c))

        # If there are interior cells, transform the component
        if interior_cells:
            # Fill interior cells based on flood-fill distance to nearest 1
            # Use BFS to calculate actual distance through cells
            distance = {}
            queue = []
            for r1, c1 in comp:
                distance[(r1, c1)] = 0
                queue.append((r1, c1))

            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) not in distance and (nr, nc) in [(r, c) for r, c in interior_cells]:
                        distance[(nr, nc)] = distance[(r, c)] + 1
                        queue.append((nr, nc))

            # Fill based on distance and neighbors
            for r, c in interior_cells:
                if (r, c) not in distance:
                    continue

                dist = distance[(r, c)]

                if dist == 1:
                    result[r][c] = 3
                elif dist == 2:
                    # Check neighbors at distance >= 2 (core region cells)
                    core_neighbors = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in distance and distance[(nr, nc)] >= 2:
                            core_neighbors += 1
                    # If has >= 2 core neighbors, it's part of a core region -> stay 0
                    # Otherwise, it's an edge cell -> fill with 3
                    if core_neighbors < 2:
                        result[r][c] = 3
                    # else: keep as 0 (part of core region)
                # else: keep as 0

            # Add border of 2s (only exterior cells adjacent to 1s or 3s)
            for r in range(max(0, min_r - 1), min(rows, max_r + 2)):
                for c in range(max(0, min_c - 1), min(cols, max_c + 2)):
                    if result[r][c] == 0 and exterior[r][c]:
                        # Check if adjacent (8-directional) to a 1 or 3
                        adjacent = False
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] in [1, 3]:
                                    adjacent = True
                                    break
                            if adjacent:
                                break
                        if adjacent:
                            result[r][c] = 2

    return result
