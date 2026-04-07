def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 0s, 4s, and 5s
    2. Output contains 0s, 2s, 4s, and 5s (some 5s convert to 2s)
    3. Pattern: Small connected components (size ≤ 2) of 5s with 2+ orthogonal 4s adjacent

    Procedure:
    1. Find connected components of 5s
    2. For components of size ≤ 2 with 2+ adjacent 4s, convert to 2
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all 4 positions
    fours = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                fours.add((r, c))

    # Find connected components of 5s
    visited = set()

    def bfs_fives(start_r, start_c):
        component = set()
        queue = [(start_r, start_c)]
        visited.add((start_r, start_c))
        component.add((start_r, start_c))

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited and grid[nr][nc] == 5:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
                        component.add((nr, nc))
        return component

    # Process each 5-component
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and (r, c) not in visited:
                component = bfs_fives(r, c)

                # Only consider small components
                if len(component) > 2:
                    continue

                # Count adjacent 4s for this component
                adjacent_fours = set()
                for cell_r, cell_c in component:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cell_r + dr, cell_c + dc
                        if (nr, nc) in fours:
                            adjacent_fours.add((nr, nc))

                # Convert if 2+ adjacent 4s
                if len(adjacent_fours) >= 2:
                    for cell_r, cell_c in component:
                        result[cell_r][cell_c] = 2

    return result
