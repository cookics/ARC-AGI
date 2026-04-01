def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has template patterns: bordered rectangles with specific interior patterns
    2. Grid has incomplete patterns to complete using templates
    3. If interior pattern matches → add border from template
    4. If uniform block matches border color and size → fill with interior pattern

    Procedure:
    1. Extract all template patterns (bordered rectangles)
    2. For each cell, try to match interior patterns and add borders
    3. For each cell, try to match uniform blocks and fill interiors
    """

    def is_background(val):
        return val in [0, 3]

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Extract templates: bordered rectangles
    templates = []

    for h in range(3, 11):
        for w in range(3, 11):
            for r in range(rows - h + 1):
                for c in range(cols - w + 1):
                    # Get border cells
                    border_cells = []
                    for dc in range(w):
                        border_cells.append(grid[r][c + dc])
                        border_cells.append(grid[r + h - 1][c + dc])
                    for dr in range(1, h - 1):
                        border_cells.append(grid[r + dr][c])
                        border_cells.append(grid[r + dr][c + w - 1])

                    # Check uniform non-background border
                    border_cells = [v for v in border_cells if not is_background(v)]
                    if not border_cells or len(set(border_cells)) != 1:
                        continue

                    border_color = border_cells[0]

                    # Get interior
                    interior = []
                    for dr in range(1, h - 1):
                        row = []
                        for dc in range(1, w - 1):
                            row.append(grid[r + dr][c + dc])
                        interior.append(row)

                    # Check interior has content and differs from border
                    has_non_bg = any(not is_background(v) for row in interior for v in row)
                    if not has_non_bg:
                        continue

                    # Interior must have values different from border color
                    has_diff = any(
                        not is_background(v) and v != border_color
                        for row in interior
                        for v in row
                    )
                    if not has_diff:
                        continue

                    templates.append({
                        'h': h,
                        'w': w,
                        'border': border_color,
                        'interior': interior,
                        'loc': (r, c)
                    })

    # Track which cells belong to templates (don't transform these)
    template_cells = set()
    for tmpl in templates:
        tr, tc = tmpl['loc']
        for dr in range(tmpl['h']):
            for dc in range(tmpl['w']):
                template_cells.add((tr + dr, tc + dc))

    # Apply transformations
    processed = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) in processed or (r, c) in template_cells:
                continue

            for tmpl in templates:
                int_h = len(tmpl['interior'])
                int_w = len(tmpl['interior'][0]) if tmpl['interior'] else 0

                # Case 1: Match interior → add border
                if r + int_h <= rows and c + int_w <= cols:
                    match = True
                    for dr in range(int_h):
                        for dc in range(int_w):
                            if grid[r + dr][c + dc] != tmpl['interior'][dr][dc]:
                                match = False
                                break
                        if not match:
                            break

                    if match:
                        # Check this interior is not part of a template
                        is_template_interior = False
                        for t in templates:
                            tr, tc = t['loc']
                            if tr < r < tr + t['h'] - 1 and tc < c < tc + t['w'] - 1:
                                is_template_interior = True
                                break

                        if not is_template_interior:
                            # Add border
                            for dr in range(tmpl['h']):
                                for dc in range(tmpl['w']):
                                    rr, cc = r - 1 + dr, c - 1 + dc
                                    if 0 <= rr < rows and 0 <= cc < cols:
                                        is_border = (dr == 0 or dr == tmpl['h'] - 1 or
                                                    dc == 0 or dc == tmpl['w'] - 1)
                                        if is_border:
                                            result[rr][cc] = tmpl['border']
                                        processed.add((rr, cc))
                            break

                # Case 2: Match uniform border block → fill interior
                if r + tmpl['h'] <= rows and c + tmpl['w'] <= cols:
                    all_border_color = True
                    for dr in range(tmpl['h']):
                        for dc in range(tmpl['w']):
                            if grid[r + dr][c + dc] != tmpl['border']:
                                all_border_color = False
                                break
                        if not all_border_color:
                            break

                    if all_border_color:
                        # Check not part of template
                        overlaps_template = any(
                            (r + dr, c + dc) in template_cells
                            for dr in range(tmpl['h'])
                            for dc in range(tmpl['w'])
                        )

                        if not overlaps_template:
                            # Fill interior
                            for dr in range(int_h):
                                for dc in range(int_w):
                                    result[r + 1 + dr][c + 1 + dc] = tmpl['interior'][dr][dc]
                                    processed.add((r + 1 + dr, c + 1 + dc))

                            # Mark all cells as processed
                            for dr in range(tmpl['h']):
                                for dc in range(tmpl['w']):
                                    processed.add((r + dr, c + dc))
                            break

    return result
