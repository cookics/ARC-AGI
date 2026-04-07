def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a horizontal line of 5s dividing top/bottom sections
    2. Input has a vertical line of 5s (in top section) dividing left/right
    3. Top-left quadrant contains reference colors (non-0, non-3, non-5)
    4. Other areas contain 3s that need to be replaced
    5. Each connected component of 3s is replaced by one color from reference set
    6. The replacement color is determined by the component size

    Procedure:
    1. Find the horizontal separator row (row with most 5s)
    2. Find the vertical separator column (column with 5s in top section)
    3. Extract reference colors from top-left quadrant (sorted)
    4. Find all connected components of 3s using BFS
    5. Replace each component with color based on formula: ref[(size - 1) % len(ref)]
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find horizontal separator (row with most 5s)
    h_sep = -1
    max_fives = 0
    for r in range(rows):
        count = sum(1 for c in range(cols) if grid[r][c] == 5)
        if count > max_fives:
            max_fives = count
            h_sep = r

    # Find vertical separator (column with 5s in top section)
    v_sep = -1
    for c in range(cols):
        if h_sep >= 0 and all(grid[r][c] == 5 for r in range(h_sep + 1)):
            v_sep = c
            break

    # Extract reference colors from top-left quadrant (sorted)
    ref_colors = set()
    if h_sep >= 0 and v_sep >= 0:
        for r in range(h_sep):
            for c in range(v_sep):
                val = grid[r][c]
                if val != 0 and val != 3 and val != 5:
                    ref_colors.add(val)

    ref_colors = sorted(list(ref_colors))

    # Create result grid (copy of input)
    result = [row[:] for row in grid]

    # Find and replace connected components of 3s
    visited = [[False] * cols for _ in range(rows)]

    def bfs(start_r, start_c):
        """Find connected component of 3s using BFS"""
        if visited[start_r][start_c] or grid[start_r][start_c] != 3:
            return []

        component = []
        queue = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.pop(0)
            component.append((r, c))

            # Check 4 adjacent cells
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if not visited[nr][nc] and grid[nr][nc] == 3:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return component

    # Collect all components with their properties
    components_info = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                component = bfs(r, c)
                if component:
                    # Calculate bounding box
                    min_r = min(cr for cr, cc in component)
                    max_r = max(cr for cr, cc in component)
                    min_c = min(cc for cr, cc in component)
                    max_c = max(cc for cr, cc in component)
                    height = max_r - min_r + 1
                    width = max_c - min_c + 1

                    components_info.append({
                        'cells': component,
                        'size': len(component),
                        'height': height,
                        'width': width,
                        'min_r': min_r,
                        'min_c': min_c
                    })

    # Group components by size
    size_groups = {}
    for comp_info in components_info:
        size = comp_info['size']
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(comp_info)

    # Assign colors
    if components_info and ref_colors:
        for size, comps in size_groups.items():
            # Base color from size
            color_idx = (size - 1) % len(ref_colors)
            base_color = ref_colors[color_idx]

            # If multiple components of same size, check if square tiebreaker applies
            if len(comps) > 1 and len(ref_colors) > 1:
                # Check if there's a mix of square and non-square bounding boxes
                has_square = any(c['height'] == c['width'] for c in comps)
                has_non_square = any(c['height'] != c['width'] for c in comps)

                if has_square and has_non_square:
                    # Apply square tiebreaker
                    alt_color_idx = (color_idx + 1) % len(ref_colors)
                    alt_color = ref_colors[alt_color_idx]

                    for comp_info in comps:
                        if comp_info['height'] == comp_info['width']:
                            color = alt_color
                        else:
                            color = base_color

                        for cr, cc in comp_info['cells']:
                            result[cr][cc] = color
                else:
                    # All same shape - use base color for all
                    for comp_info in comps:
                        for cr, cc in comp_info['cells']:
                            result[cr][cc] = base_color
            else:
                # Single component of this size - use base color
                for comp_info in comps:
                    for cr, cc in comp_info['cells']:
                        result[cr][cc] = base_color

    return result
