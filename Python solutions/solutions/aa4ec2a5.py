def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background 4 and connected regions of 1s
    2. Output transforms each connected component by adding border of 2s
    3. Interior filling depends on whether component has gaps in its bounding box:
       - No gaps (solid rectangle): keep 1s
       - Has gaps: convert 1s→8s, internal gaps→6s, internal borders→2
    4. Gap is a 4 within the component's bounding box not part of the component

    Procedure:
    1. Find all connected components of 1s using DFS/BFS
    2. For each component:
       a. Calculate bounding box
       b. Determine if component has gaps
       c. Add outer border of 2s
       d. Fill interior based on gap status
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(start_r, start_c):
        q = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = set([(start_r, start_c)])

        while q:
            r, c = q.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 1:
                    visited[nr][nc] = True
                    q.append((nr, nc))
                    cells.add((nr, nc))

        return cells

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                component = bfs(r, c)
                components.append(component)

    # Process each component
    for component in components:
        component_set = set(component)

        # Get bounding box
        min_r = min(r for r, c in component_set)
        max_r = max(r for r, c in component_set)
        min_c = min(c for r, c in component_set)
        max_c = max(c for r, c in component_set)

        # Use flood fill to find "exposed" non-component cells (reachable from bounding box edge)
        exposed = set()
        q = deque()

        # Start flood fill from edges of bounding box
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in component_set:
                    # Check if at edge of bounding box
                    if r == min_r or r == max_r or c == min_c or c == max_c:
                        if (r, c) not in exposed:
                            q.append((r, c))
                            exposed.add((r, c))

        # BFS through non-component cells within bounding box
        while q:
            r, c = q.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if (min_r <= nr <= max_r and min_c <= nc <= max_c and
                    (nr, nc) not in component_set and (nr, nc) not in exposed):
                    exposed.add((nr, nc))
                    q.append((nr, nc))

        # Check if there are any enclosed holes (non-exposed non-component cells)
        has_enclosed_holes = False
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in component_set and (r, c) not in exposed:
                    has_enclosed_holes = True
                    break
            if has_enclosed_holes:
                break

        # Find all cells adjacent to component (including diagonals for border detection)
        adjacent_to_component = set()
        for r, c in component_set:
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in component_set:
                    adjacent_to_component.add((nr, nc))

        if not has_enclosed_holes:
            # Solid component (no enclosed holes): border all adjacent cells, keep interior as 1s
            for r, c in adjacent_to_component:
                result[r][c] = 2
            for r, c in component_set:
                result[r][c] = 1
        else:
            # Component with enclosed holes - need to distinguish borders from holes
            # Mark all adjacent cells outside bounding box as borders
            for r, c in adjacent_to_component:
                if r < min_r or r > max_r or c < min_c or c > max_c:
                    result[r][c] = 2

            # Fill bounding box interior
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if (r, c) in component_set:
                        result[r][c] = 8
                    elif (r, c) in exposed:
                        # Exposed non-component cells adjacent to component are borders
                        if (r, c) in adjacent_to_component:
                            result[r][c] = 2
                        # Exposed non-component cells not adjacent stay as current value
                    else:
                        # Enclosed cells (holes)
                        result[r][c] = 6

    return result
