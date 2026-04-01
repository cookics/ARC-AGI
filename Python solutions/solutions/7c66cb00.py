def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid has horizontal sections with borders and fills
    2. Background section contains colored patterns
    3. Patterns are copied to sections where pattern color matches section fill
    4. Pattern cells map: boundary → section border, interior → section fill
    5. Column positions preserved, patterns placed at bottom of sections

    Procedure:
    1. Identify bordered sections (containers) and background rows
    2. Extract colored patterns from background (connected components)
    3. For each container, place matching patterns preserving columns
    4. Map colors: boundary cells → border, interior cells → fill
    """
    from collections import Counter

    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find bordered sections (containers)
    containers = []
    r = 0
    while r < h:
        if len(set(grid[r])) == 2:
            border, fill = grid[r][0], grid[r][1]
            # Check if valid bordered row
            if border != fill:
                is_bordered = all(
                    grid[r][i] == (border if i in [0, w-1] else fill)
                    for i in range(w)
                )
                if is_bordered:
                    start = r
                    # Find end of this container
                    while r < h and len(set(grid[r])) == 2:
                        if grid[r][0] == border and grid[r][-1] == border:
                            r += 1
                        else:
                            break
                    containers.append((start, r - 1, border, fill))
                    continue
        r += 1

    # Find background rows (not in containers)
    container_rows = set()
    for s, e, _, _ in containers:
        container_rows.update(range(s, e + 1))
    bg_rows = sorted(r for r in range(h) if r not in container_rows)

    if not bg_rows:
        return result

    # Determine background color (most common in background rows)
    bg_count = Counter()
    for r in bg_rows:
        bg_count.update(grid[r])
    bg_color = bg_count.most_common(1)[0][0]

    # Extract connected components from background
    visited = set()
    components = []

    for r in bg_rows:
        for c in range(w):
            if (r, c) not in visited and grid[r][c] != bg_color:
                # BFS to find connected component
                component = []
                queue = [(r, c)]
                visited.add((r, c))
                comp_colors = set()

                while queue:
                    cr, cc = queue.pop(0)
                    component.append((cr, cc, grid[cr][cc]))
                    comp_colors.add(grid[cr][cc])

                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                            if nr in bg_rows and grid[nr][nc] != bg_color:
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                components.append((component, comp_colors))

    # Clear background
    for r in bg_rows:
        result[r] = [bg_color] * w

    # Place components into matching containers
    for start, end, border, fill in containers:
        for component, comp_colors in components:
            # Check if this component contains the fill color
            if fill not in comp_colors:
                continue

            # Create set of component cells for fast lookup
            comp_cells = {(r, c) for r, c, v in component}

            # Group cells by row
            row_groups = {}
            for r, c, v in component:
                if r not in row_groups:
                    row_groups[r] = []
                row_groups[r].append(c)

            sorted_rows = sorted(row_groups.keys())
            pattern_height = len(sorted_rows)

            # Map to bottom of container
            for i, src_row in enumerate(sorted_rows):
                target_row = end - pattern_height + 1 + i

                if start <= target_row <= end:
                    cols = row_groups[src_row]

                    for c in cols:
                        if 1 <= c < w - 1:
                            # Check if this cell is on boundary (has background or out-of-component neighbor)
                            is_boundary = False
                            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                                nr, nc = src_row + dr, c + dc
                                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                                    is_boundary = True
                                    break
                                if (nr, nc) not in comp_cells:
                                    # Neighbor is not in component
                                    is_boundary = True
                                    break

                            result[target_row][c] = border if is_boundary else fill

    return result
