def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains L-shaped patterns (horizontal line + vertical extension)
    2. Each pattern has a primary color and a unique marker cell inside
    3. If the marker appears standalone elsewhere, the pattern moves to align with it
    4. Patterns move so that: starting row = standalone row, vertical column = standalone column

    Procedure:
    1. Identify all L-shaped patterns with their markers
    2. Find standalone marker cells
    3. Move patterns with standalone markers; keep others in place
    4. Remaining standalone cells stay in original positions
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all L-shaped patterns (both normal and inverted)
    patterns = []
    used_cells = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) in used_cells or grid[r][c] == 0:
                continue

            color = grid[r][c]

            # Type 1: Horizontal at top, vertical extending down
            if r + 2 < rows:
                h_len = 0
                while c + h_len < cols and grid[r][c + h_len] == color:
                    h_len += 1

                if h_len >= 2:
                    v_col = None
                    # Check left end
                    if (c < cols and r + 1 < rows and r + 2 < rows and
                        grid[r + 1][c] == color and grid[r + 2][c] == color):
                        v_col = c
                    # Check right end
                    elif (r + 1 < rows and r + 2 < rows and c + h_len - 1 < cols and
                          grid[r + 1][c + h_len - 1] == color and
                          grid[r + 2][c + h_len - 1] == color):
                        v_col = c + h_len - 1

                    if v_col is not None:
                        # Found L-pattern (type 1)
                        pattern_cells = []

                        # Horizontal line
                        for cc in range(c, c + h_len):
                            pattern_cells.append((r, cc, grid[r][cc]))
                            used_cells.add((r, cc))

                        # Vertical extension
                        for rr in range(r + 1, r + 3):
                            if rr < rows:
                                pattern_cells.append((rr, v_col, grid[rr][v_col]))
                                used_cells.add((rr, v_col))

                        # Look for marker
                        marker_value = None
                        marker_pos = None

                        for rr in range(r, min(r + 3, rows)):
                            for cc in range(c, c + h_len):
                                if (rr, cc) not in used_cells and grid[rr][cc] != 0:
                                    marker_value = grid[rr][cc]
                                    marker_pos = (rr, cc)
                                    pattern_cells.append((rr, cc, marker_value))
                                    used_cells.add((rr, cc))

                        patterns.append({
                            'color': color,
                            'cells': pattern_cells,
                            'marker': marker_value,
                            'marker_pos': marker_pos,
                            'start_row': r,
                            'start_col': c,
                            'h_len': h_len,
                            'v_col': v_col,
                            'type': 'normal'
                        })
                        continue

            # Type 2: Vertical at top, horizontal at bottom
            if r + 2 < rows and grid[r + 1][c] == color and grid[r + 2][c] == color:
                # Found vertical line, check for horizontal at bottom
                h_len = 0
                while c + h_len < cols and grid[r + 2][c + h_len] == color:
                    h_len += 1

                if h_len >= 2:
                    # Found inverted L-pattern
                    pattern_cells = []

                    # Vertical line
                    for rr in range(r, r + 3):
                        if rr < rows:
                            pattern_cells.append((rr, c, grid[rr][c]))
                            used_cells.add((rr, c))

                    # Horizontal line (excluding the corner which is already added)
                    for cc in range(c + 1, c + h_len):
                        pattern_cells.append((r + 2, cc, grid[r + 2][cc]))
                        used_cells.add((r + 2, cc))

                    # Look for marker
                    marker_value = None
                    marker_pos = None

                    for rr in range(r, min(r + 3, rows)):
                        for cc in range(c, c + h_len):
                            if (rr, cc) not in used_cells and grid[rr][cc] != 0:
                                marker_value = grid[rr][cc]
                                marker_pos = (rr, cc)
                                pattern_cells.append((rr, cc, marker_value))
                                used_cells.add((rr, cc))

                    patterns.append({
                        'color': color,
                        'cells': pattern_cells,
                        'marker': marker_value,
                        'marker_pos': marker_pos,
                        'start_row': r,
                        'start_col': c,
                        'h_len': h_len,
                        'v_col': c,
                        'type': 'inverted'
                    })

    # Find standalone marker cells (not part of any pattern)
    standalone_markers = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in used_cells and grid[r][c] != 0:
                val = grid[r][c]
                if val not in standalone_markers:
                    standalone_markers[val] = []
                standalone_markers[val].append((r, c))

    # Process patterns: move if marker has standalone, else keep in place
    for pattern in patterns:
        marker = pattern['marker']

        if marker and marker in standalone_markers and len(standalone_markers[marker]) > 0:
            # Move pattern to align with standalone marker
            target_r, target_c = standalone_markers[marker][0]

            # Calculate offset
            old_start_r = pattern['start_row']
            old_v_col = pattern['v_col']

            # New position: start_row = target_r, v_col = target_c
            row_offset = target_r - old_start_r
            col_offset = target_c - old_v_col

            # Place moved pattern
            for r, c, val in pattern['cells']:
                new_r = r + row_offset
                new_c = c + col_offset
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    result[new_r][new_c] = val
        else:
            # Keep pattern in original position
            for r, c, val in pattern['cells']:
                result[r][c] = val

    # Place remaining standalone cells
    for val, positions in standalone_markers.items():
        # Check if this marker was used for pattern movement
        used_for_movement = False
        for pattern in patterns:
            if pattern['marker'] == val:
                used_for_movement = True
                break

        if not used_for_movement:
            for r, c in positions:
                result[r][c] = val

    return result
