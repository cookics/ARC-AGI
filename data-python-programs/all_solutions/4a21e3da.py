def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Markers (2s) at edges define dividing lines (vertical/horizontal)
    2. Dividing lines extend from marker to where 7s pattern touches the line
    3. 7s on dividing lines stay in place
    4. 7s not on lines: split into regions, move to corners maintaining relative positions

    Procedure:
    1. Find markers and determine dividing line positions
    2. Find extent of dividing lines (from marker to 7s pattern)
    3. Draw dividing lines with 2s (except where input has 7s)
    4. Transform 7s: keep those on lines, move others to corners by region
    """
    height = len(grid)
    width = len(grid[0])

    markers = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == 2]
    sevens = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == 7]

    if not sevens:
        return [row[:] for row in grid]

    min_r = min(r for r, _ in sevens)
    max_r = max(r for r, _ in sevens)
    min_c = min(c for _, c in sevens)
    max_c = max(c for _, c in sevens)

    # Determine dividing lines
    v_line = None
    h_line = None
    for r, c in markers:
        if r == 0 or r == height - 1:
            v_line = c
        if c == 0 or c == width - 1:
            h_line = r

    result = [[1] * width for _ in range(height)]

    # Find extent of dividing lines based on where 7s touch them
    if v_line is not None:
        rows_with_7_at_vline = [r for r, c in sevens if c == v_line]
        if rows_with_7_at_vline:
            v_line_start = min(rows_with_7_at_vline)
            v_line_end = max(rows_with_7_at_vline)
        else:
            v_line_start = min_r
            v_line_end = max_r

        # Extend to marker
        for r, c in markers:
            if c == v_line:
                if r == 0:
                    v_line_start = min(v_line_start, 0)
                elif r == height - 1:
                    v_line_end = max(v_line_end, height - 1)

    if h_line is not None:
        cols_with_7_at_hline = [c for r, c in sevens if r == h_line]
        if cols_with_7_at_hline:
            h_line_start = min(cols_with_7_at_hline)
            h_line_end = max(cols_with_7_at_hline)
        else:
            h_line_start = min_c
            h_line_end = max_c

        # Extend to marker
        for r, c in markers:
            if r == h_line:
                if c == 0:
                    h_line_start = min(h_line_start, 0)
                elif c == width - 1:
                    h_line_end = max(h_line_end, width - 1)

    # Draw dividing lines with 2s
    if v_line is not None:
        for r in range(v_line_start, v_line_end + 1):
            if grid[r][v_line] != 7:
                result[r][v_line] = 2

    if h_line is not None:
        for c in range(h_line_start, h_line_end + 1):
            if grid[h_line][c] != 7:
                result[h_line][c] = 2

    # Place 7s on dividing lines (they stay in place)
    for r, c in sevens:
        if v_line is not None and c == v_line:
            result[r][v_line] = 7
        if h_line is not None and r == h_line:
            result[h_line][c] = 7

    # Transform 7s not on dividing lines - need separate bbox per quadrant
    if v_line is not None and h_line is not None:
        # Both lines: split into 4 quadrants, each with its own bbox
        # Only process quadrants where dividing lines extend to the edge
        quadrants = {
            'tl': ([(r, c) for r, c in sevens if r < h_line and c < v_line],
                   v_line_start == 0 or h_line_start == 0),
            'tr': ([(r, c) for r, c in sevens if r < h_line and c > v_line],
                   v_line_start == 0 or h_line_end == width - 1),
            'bl': ([(r, c) for r, c in sevens if r > h_line and c < v_line],
                   v_line_end == height - 1 or h_line_start == 0),
            'br': ([(r, c) for r, c in sevens if r > h_line and c > v_line],
                   v_line_end == height - 1 or h_line_end == width - 1)
        }

        for quad_name, (quad_sevens, is_active) in quadrants.items():
            if not quad_sevens or not is_active:
                continue

            q_min_r = min(r for r, _ in quad_sevens)
            q_max_r = max(r for r, _ in quad_sevens)
            q_min_c = min(c for _, c in quad_sevens)
            q_max_c = max(c for _, c in quad_sevens)

            for r, c in quad_sevens:
                r_off = r - q_min_r
                c_off = c - q_min_c

                if quad_name == 'tl':
                    result[r_off][c_off] = 7
                elif quad_name == 'tr':
                    new_c = width - 1 - (q_max_c - c)
                    result[r_off][new_c] = 7
                elif quad_name == 'bl':
                    new_r = height - 1 - (q_max_r - r)
                    result[new_r][c_off] = 7
                elif quad_name == 'br':
                    new_r = height - 1 - (q_max_r - r)
                    new_c = width - 1 - (q_max_c - c)
                    result[new_r][new_c] = 7

    elif v_line is not None:
        # Only vertical line: split left/right
        left_sevens = [(r, c) for r, c in sevens if c < v_line]
        right_sevens = [(r, c) for r, c in sevens if c > v_line]

        marker_r = markers[0][0]

        for side_sevens in [left_sevens, right_sevens]:
            if not side_sevens:
                continue

            s_min_r = min(r for r, _ in side_sevens)
            s_max_r = max(r for r, _ in side_sevens)
            s_min_c = min(c for _, c in side_sevens)
            s_max_c = max(c for _, c in side_sevens)

            for r, c in side_sevens:
                r_off = r - s_min_r
                c_off = c - s_min_c

                if marker_r == 0:  # Top marker
                    new_r = r_off
                else:  # Bottom marker
                    new_r = height - 1 - (s_max_r - r)

                if c < v_line:  # Left side
                    new_c = c_off
                else:  # Right side
                    new_c = width - 1 - (s_max_c - c)

                result[new_r][new_c] = 7

    elif h_line is not None:
        # Only horizontal line: split top/bottom
        top_sevens = [(r, c) for r, c in sevens if r < h_line]
        bottom_sevens = [(r, c) for r, c in sevens if r > h_line]

        marker_c = markers[0][1]

        for side_sevens in [top_sevens, bottom_sevens]:
            if not side_sevens:
                continue

            s_min_r = min(r for r, _ in side_sevens)
            s_max_r = max(r for r, _ in side_sevens)
            s_min_c = min(c for _, c in side_sevens)
            s_max_c = max(c for _, c in side_sevens)

            for r, c in side_sevens:
                r_off = r - s_min_r
                c_off = c - s_min_c

                if marker_c == 0:  # Left marker
                    new_c = c_off
                else:  # Right marker
                    new_c = width - 1 - (s_max_c - c)

                if r < h_line:  # Top side
                    new_r = r_off
                else:  # Bottom side
                    new_r = height - 1 - (s_max_r - r)

                result[new_r][new_c] = 7

    return result
