def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 12x12 grid with two patterns: 1s (top) and 5s (bottom)
    2. Output keeps 1s unchanged, some 5s stay as 5, others become 2
    3. Connected components of 5s are scored based on sum of 1s in their columns
    4. Selection rule depends on minimum sum and distribution gap

    Procedure:
    1. Find connected components of 5s
    2. Score each by sum of 1s in the corresponding columns
    3. Apply adaptive selection based on min sum and gap to next sum
    4. Convert non-selected 5s to 2s
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find rows with 1s
    rows_with_ones = []
    for r in range(rows):
        if any(grid[r][c] == 1 for c in range(cols)):
            rows_with_ones.append(r)

    if not rows_with_ones:
        return result

    # Find connected components of 5s
    def find_5s_components():
        visited = [[False] * cols for _ in range(rows)]
        components = []

        def bfs(start_r, start_c):
            queue = [(start_r, start_c)]
            visited[start_r][start_c] = True
            cells = []
            comp_cols = set()

            while queue:
                r, c = queue.pop(0)
                cells.append((r, c))
                comp_cols.add(c)

                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr][nc] and grid[nr][nc] == 5:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

            return cells, comp_cols

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 5 and not visited[r][c]:
                    cells, comp_cols = bfs(r, c)
                    components.append((cells, comp_cols))

        return components

    components = find_5s_components()
    if not components:
        return result

    # Calculate sum of 1s for each column
    col_sums = [0] * cols
    for r in rows_with_ones:
        for c in range(cols):
            if grid[r][c] == 1:
                col_sums[c] += 1

    # Score each component
    component_scores = []
    for cells, comp_cols in components:
        total_sum = sum(col_sums[c] for c in comp_cols)
        min_col = min(comp_cols) if comp_cols else 0
        max_col = max(comp_cols) if comp_cols else 0
        center = (min_col + max_col) / 2
        dist_from_center = abs(center - (cols / 2))

        component_scores.append({
            'cells': cells,
            'total_sum': total_sum,
            'dist_from_center': dist_from_center,
            'min_col': min_col
        })

    # Sort by sum (asc), distance from center (asc), min_col (asc)
    component_scores.sort(key=lambda x: (x['total_sum'], x['dist_from_center'], x['min_col']))

    # Determine which components to preserve
    preserved_cells = set()
    if component_scores:
        min_sum = component_scores[0]['total_sum']
        all_sums = sorted(set(c['total_sum'] for c in component_scores))

        # Check if we should apply tolerance
        if len(all_sums) >= 2 and min_sum == 1:
            # Check gap to next sum
            next_sum = all_sums[1]
            gap = next_sum - min_sum

            if gap >= 2:
                # Large gap: apply tolerance
                for comp in component_scores:
                    if comp['total_sum'] <= min_sum + 3:
                        preserved_cells.update(comp['cells'])
            else:
                # Small gap: keep only minimum
                preserved_cells.update(component_scores[0]['cells'])
        else:
            # Default: keep only the best component
            preserved_cells.update(component_scores[0]['cells'])

    # Convert non-preserved 5s to 2s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                if (r, c) not in preserved_cells:
                    result[r][c] = 2

    return result
