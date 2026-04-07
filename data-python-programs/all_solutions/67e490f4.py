def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with a large intricate rectangular template and scattered shapes
    2. The template is a region with a frame color and holes (different color inside)
    3. Scattered shapes outside the template have various colors and shapes (2x2 blocks, plus shapes, lines)
    4. Output extracts the template and creates a 4-fold symmetric pattern by filling holes with colors
    5. Colors are assigned based on shape matching: 2x2 blocks to holes, plus shapes to center, etc.

    Procedure:
    1. Find the large rectangular template (dominant non-background color forming a grid)
    2. Extract the template
    3. Find scattered colored shapes outside the template
    4. Classify shapes by type (2x2 block, plus/cross, line, single)
    5. Create symmetric pattern: match hole shapes to scattered shape colors
    6. The matching respects symmetry: symmetric holes get the same color
    """
    from collections import deque, Counter

    rows, cols = len(grid), len(grid[0])

    # Find background color (most common)
    color_counts = Counter()
    for r in range(rows):
        for c in range(cols):
            color_counts[grid[r][c]] += 1

    background = color_counts.most_common(1)[0][0]

    # Find the template region (large rectangular region with intricate pattern)
    def find_template():
        best = None
        best_score = 0

        for template_color in range(10):
            if template_color == background:
                continue

            # Find all positions with this color
            positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == template_color]
            if not positions:
                continue

            min_r = min(r for r, c in positions)
            max_r = max(r for r, c in positions)
            min_c = min(c for r, c in positions)
            max_c = max(c for r, c in positions)

            height = max_r - min_r + 1
            width = max_c - min_c + 1

            # Must be reasonably large (at least 5x5)
            if height < 5 or width < 5:
                continue

            # Skip if unreasonably large (larger than input)
            if height > rows or width > cols:
                continue

            # Check if border is all template_color (full frame)
            is_frame = True
            for c in range(min_c, max_c + 1):
                if grid[min_r][c] != template_color or grid[max_r][c] != template_color:
                    is_frame = False
                    break
            if is_frame:
                for r in range(min_r + 1, max_r):
                    if grid[r][min_c] != template_color or grid[r][max_c] != template_color:
                        is_frame = False
                        break

            if not is_frame:
                continue

            # Count color distribution inside
            interior_colors = set()
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    interior_colors.add(grid[r][c])

            # Should have at least 2 colors (frame + holes)
            if len(interior_colors) < 2:
                continue

            # Prefer more square-like templates and larger sizes
            squareness = min(height, width) / max(height, width)
            score = height * width * (1 + squareness)  # Bonus for being square

            if score > best_score:
                best_score = score
                best = (min_r, max_r, min_c, max_c, template_color)

        return best

    template_info = find_template()
    if not template_info:
        # Fallback: if no clear template, try first non-background color with largest footprint
        return grid  # Return input as-is as last resort

    t_r1, t_r2, t_c1, t_c2, template_color = template_info
    t_height = t_r2 - t_r1 + 1
    t_width = t_c2 - t_c1 + 1

    # Extract template
    template = []
    for r in range(t_r1, t_r2 + 1):
        template.append(grid[r][t_c1:t_c2 + 1])

    # Find scattered shapes outside template
    def find_components_outside():
        visited = set()
        components = []

        for r in range(rows):
            for c in range(cols):
                # Skip template region
                if t_r1 <= r <= t_r2 and t_c1 <= c <= t_c2:
                    continue

                if (r, c) in visited or grid[r][c] == background:
                    continue

                # BFS to find component
                color = grid[r][c]
                queue = deque([(r, c)])
                visited.add((r, c))
                component = [(r, c)]

                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                            if not (t_r1 <= nr <= t_r2 and t_c1 <= nc <= t_c2):
                                if grid[nr][nc] == color:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                                    component.append((nr, nc))

                components.append((color, component))

        return components

    scattered = find_components_outside()

    # Classify scattered shapes
    def classify_shape(component):
        if len(component) == 1:
            return 'single'

        # Check for lines (any length)
        rs = [r for r, c in component]
        cs = [c for r, c in component]

        # Vertical line check
        if len(set(cs)) == 1:
            # All same column
            sorted_rs = sorted(rs)
            if all(sorted_rs[i] + 1 == sorted_rs[i+1] for i in range(len(sorted_rs) - 1)):
                # Continuous vertical line
                if len(component) == 2:
                    return 'v_line2'
                elif len(component) == 3:
                    return 'v_line3'
                else:
                    return 'v_line_long'

        # Horizontal line check
        if len(set(rs)) == 1:
            # All same row
            sorted_cs = sorted(cs)
            if all(sorted_cs[i] + 1 == sorted_cs[i+1] for i in range(len(sorted_cs) - 1)):
                # Continuous horizontal line
                if len(component) == 2:
                    return 'h_line2'
                elif len(component) == 3:
                    return 'h_line3'
                else:
                    return 'h_line_long'

        # 2x2 block check
        if len(component) == 4:
            rs_sorted = sorted(rs)
            cs_sorted = sorted(cs)
            if rs_sorted[0] == rs_sorted[1] and rs_sorted[2] == rs_sorted[3] and rs_sorted[2] == rs_sorted[0] + 1:
                if cs_sorted[0] == cs_sorted[2] and cs_sorted[1] == cs_sorted[3] and cs_sorted[1] == cs_sorted[0] + 1:
                    return 'block2x2'

        # Plus shape check
        if len(component) == 5:
            center_r = sum(rs) // 5
            center_c = sum(cs) // 5
            expected = {(center_r, center_c), (center_r-1, center_c), (center_r+1, center_c),
                       (center_r, center_c-1), (center_r, center_c+1)}
            if set(component) == expected:
                return 'plus'

        # L-shape or T-shape check (3 cells)
        if len(component) == 3:
            # Check for L-shape patterns
            return 'lshape3'

        return 'other'

    # Count each (color, shape_type) combination
    color_shape_counts = {}
    for color, comp in scattered:
        shape_type = classify_shape(comp)
        if color not in color_shape_counts:
            color_shape_counts[color] = {}
        color_shape_counts[color][shape_type] = color_shape_counts[color].get(shape_type, 0) + 1

    # For each shape type, find the color with highest count
    # Also group similar shapes (lines together, blocks together)
    def normalize_shape(shape_type):
        if shape_type in ['v_line2', 'h_line2']:
            return 'line2'
        if shape_type in ['v_line3', 'h_line3', 'v_line_long', 'h_line_long', 'lshape3']:
            return 'line_or_lshape'
        return shape_type

    # Build normalized shape counts
    color_norm_shape_counts = {}
    for color, shape_counts in color_shape_counts.items():
        color_norm_shape_counts[color] = {}
        for shape_type, count in shape_counts.items():
            norm_shape = normalize_shape(shape_type)
            color_norm_shape_counts[color][norm_shape] = color_norm_shape_counts[color].get(norm_shape, 0) + count

    # Find best color for each shape type
    shape_color_map = {}
    all_shape_types = set()
    for color_counts in color_shape_counts.values():
        all_shape_types.update(color_counts.keys())

    for shape_type in all_shape_types:
        norm_shape = normalize_shape(shape_type)
        best_color = None
        best_count = 0
        best_purity = 0

        for color, norm_counts in color_norm_shape_counts.items():
            count = norm_counts.get(norm_shape, 0)
            # Calculate purity: ratio of this shape to total shapes for this color
            total = sum(norm_counts.values())
            purity = count / total if total > 0 else 0

            # Prefer higher count, then higher purity
            if count > best_count or (count == best_count and purity > best_purity):
                best_count = count
                best_purity = purity
                best_color = color

        if best_color:
            shape_color_map[shape_type] = best_color

    # Find holes in template and classify them
    def find_holes():
        visited = [[False] * t_width for _ in range(t_height)]
        holes = []

        for r in range(t_height):
            for c in range(t_width):
                if visited[r][c] or template[r][c] == template_color:
                    continue

                # BFS to find hole
                queue = deque([(r, c)])
                visited[r][c] = True
                hole = [(r, c)]

                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < t_height and 0 <= nc < t_width and not visited[nr][nc]:
                            if template[nr][nc] != template_color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                                hole.append((nr, nc))

                holes.append(hole)

        return holes

    holes = find_holes()

    # Create output starting with template
    output = [row[:] for row in template]

    # Fill holes with colors based on shape matching
    for hole in holes:
        shape_type = classify_shape(hole)
        fill_color = shape_color_map.get(shape_type, template_color)

        for r, c in hole:
            output[r][c] = fill_color

    return output
