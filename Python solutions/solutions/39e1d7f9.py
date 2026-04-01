def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided by separator lines (full rows/cols of same value)
    2. Some cells contain colored blocks (non-zero, non-separator values)
    3. One "anchor" value appears multiple times
    4. One anchor has a complete pattern (cross or 3x3) of "fill" values around it
    5. This pattern should be replicated around all other anchors

    Procedure:
    1. Parse grid into cells based on separator lines
    2. Find filled cells and identify anchor/fill values
    3. Find template pattern (anchor with complete surrounding pattern)
    4. Replicate template pattern around all anchors
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find separator color
    separator_color = None
    for color in range(1, 10):
        h_lines = [
            r for r in range(rows) if all(grid[r][c] == color for c in range(cols))
        ]
        v_lines = [
            c for c in range(cols) if all(grid[r][c] == color for r in range(rows))
        ]

        if len(h_lines) > 0 and len(v_lines) > 0:
            separator_color = color
            break

    if separator_color is None:
        return result

    # Find cell boundaries
    h_separators = [
        r
        for r in range(rows)
        if all(grid[r][c] == separator_color for c in range(cols))
    ]
    v_separators = [
        c
        for c in range(cols)
        if all(grid[r][c] == separator_color for r in range(rows))
    ]

    h_boundaries = [-1] + h_separators + [rows]
    v_boundaries = [-1] + v_separators + [cols]

    n_cell_rows = len(h_boundaries) - 1
    n_cell_cols = len(v_boundaries) - 1

    # Find cells that contain patterns
    filled_cells = {}
    for cell_r in range(n_cell_rows):
        for cell_c in range(n_cell_cols):
            start_r = h_boundaries[cell_r] + 1
            end_r = h_boundaries[cell_r + 1]
            start_c = v_boundaries[cell_c] + 1
            end_c = v_boundaries[cell_c + 1]

            # Check if this cell contains any non-zero, non-separator pattern
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    if grid[r][c] != 0 and grid[r][c] != separator_color:
                        filled_cells[(cell_r, cell_c)] = grid[r][c]
                        break
                else:
                    continue
                break

    if not filled_cells:
        return result

    # Find anchor value and fill value
    # The anchor is the value that should have patterns around it
    # Try to find a template: an anchor with a pattern around it
    anchor_val = None
    fill_val = None
    template_pattern = {}  # Relative positions (dr, dc) -> value
    template_center = None

    # Check each filled cell to see if it has a pattern around it
    # Prioritize templates with a single fill value and more neighbors
    best_template = None
    best_score = 0

    for (ci, cj), val in filled_cells.items():
        neighbors_8 = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        # Check for pattern around this cell
        pattern = {}
        fill_candidates = {}

        for dr, dc in neighbors_8:
            ni, nj = ci + dr, cj + dc
            if (ni, nj) in filled_cells:
                pattern[(dr, dc)] = filled_cells[(ni, nj)]
                if filled_cells[(ni, nj)] != val:
                    fill_val = filled_cells[(ni, nj)]
                    fill_candidates[fill_val] = fill_candidates.get(fill_val, 0) + 1

        # Score this template: prefer single fill value with many neighbors
        if len(pattern) >= 3 and len(fill_candidates) > 0:
            # Find most common fill value
            most_common_fill = max(fill_candidates.items(), key=lambda x: x[1])
            fill_val = most_common_fill[0]
            fill_count = most_common_fill[1]

            # Filter pattern to only include this fill value
            filtered_pattern = {pos: pval for pos, pval in pattern.items()
                                if pval == fill_val or pval == val}

            # Score: prefer templates with single fill type and more neighbors
            score = fill_count * 10 - len(fill_candidates)

            if score > best_score:
                best_score = score
                anchor_val = val
                best_fill_val = fill_val
                template_pattern = filtered_pattern
                template_center = (ci, cj)
                best_template = (anchor_val, best_fill_val, template_pattern, template_center)

    if best_template:
        anchor_val, fill_val, template_pattern, template_center = best_template

    if anchor_val is None:
        return result

    # Apply template pattern to all anchors
    anchors = [(ci, cj) for (ci, cj), val in filled_cells.items() if val == anchor_val]

    for ci, cj in anchors:
        # Apply template pattern around this anchor
        for (dr, dc), pattern_val in template_pattern.items():
            ni, nj = ci + dr, cj + dc
            if 0 <= ni < n_cell_rows and 0 <= nj < n_cell_cols:
                filled_cells[(ni, nj)] = pattern_val

    # Apply patterns to result grid
    for (cell_r, cell_c), color in filled_cells.items():
        start_r = h_boundaries[cell_r] + 1
        end_r = h_boundaries[cell_r + 1]
        start_c = v_boundaries[cell_c] + 1
        end_c = v_boundaries[cell_c + 1]

        # Fill the entire cell with the color
        for r in range(start_r, end_r):
            for c in range(start_c, end_c):
                result[r][c] = color

    return result
