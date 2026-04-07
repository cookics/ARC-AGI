def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has horizontal lines of 7s scattered across rows
    2. Input has 6 markers (usually at bottom row)
    3. Single-cell gaps (0 between 7s) in horizontal lines get filled with 6
    4. Vertical lines of 2s extend from 6 markers
    5. When vertical line intersects horizontal line:
       - 7 at intersection → 8
       - Cap pattern on the "source" side: 4 (intersection col), 2s (filling), 3 (right edge+1)
    6. New vertical lines spawn from cap edges continuing in SAME direction
    7. Place 6s at grid edges where spawned upward lines terminate

    Procedure:
    1. Find horizontal lines, fill single-cell gaps with 6
    2. From bottom markers (6s at max row), extend upward
    3. When hitting intersection: create 8, create cap below, DON'T continue past
    4. From cap right edge, spawn new line continuing upward
    5. Spawned lines may create their own intersections/caps recursively
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find horizontal line segments (continuous 7s only)
    h_lines = []
    for r in range(rows):
        c = 0
        while c < cols:
            if result[r][c] == 7:
                start = c
                while c < cols and result[r][c] == 7:
                    c += 1
                end = c - 1
                h_lines.append((r, start, end))
            else:
                c += 1

    # Build a lookup for gaps between segments (for later vertical line processing)
    line_gaps = {}  # {(r, c): (start, end)} - maps gap position to merged line range
    lines_by_row = {}
    for r, start, end in h_lines:
        if r not in lines_by_row:
            lines_by_row[r] = []
        lines_by_row[r].append((start, end))

    for r in lines_by_row:
        segments = sorted(lines_by_row[r])
        for i in range(len(segments) - 1):
            end1 = segments[i][1]
            start2 = segments[i + 1][0]
            # Check if there's exactly one cell gap
            if start2 - end1 == 2:
                gap_col = end1 + 1
                seg1_len = segments[i][1] - segments[i][0] + 1
                seg2_len = segments[i + 1][1] - segments[i + 1][0] + 1
                # Track gaps that could be filled (if a vertical line passes through)
                if seg1_len + seg2_len + 1 <= 7:
                    line_gaps[(r, gap_col)] = (segments[i][0], segments[i + 1][1])
                    # Add merged line to h_lines for intersection detection
                    h_lines.append((r, segments[i][0], segments[i + 1][1]))

    # Find all markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 6:
                markers.append((r, c))

    if not markers:
        return result

    # Find bottom markers
    max_row = max(r for r, c in markers)

    from collections import deque

    queue = deque()
    for r, c in markers:
        if r == max_row:
            queue.append((r, c, "up"))

    processed = set()

    while queue:
        start_r, col, direction = queue.popleft()

        if (start_r, col, direction) in processed:
            continue
        processed.add((start_r, col, direction))

        step = -1 if direction == "up" else 1
        r = start_r + step

        # Draw line until intersection or boundary
        while 0 <= r < rows:
            # Check for intersection
            intersects = None
            for hr, hstart, hend in h_lines:
                if hr == r and hstart <= col <= hend:
                    intersects = (hr, hstart, hend)
                    break

            if intersects:
                hr, hstart, hend = intersects

                # Check if this is actually a gap in a merged line
                if result[hr][col] == 0 and (hr, col) in line_gaps:
                    # This is a gap - mark it as 6 and stop
                    result[hr][col] = 6
                    break
                elif result[hr][col] == 7:
                    # Real intersection with a 7
                    result[hr][col] = 8

                    # Create cap BEFORE intersection (in direction we came from)
                    cap_r = hr - step
                    if 0 <= cap_r < rows and cap_r != start_r:
                        result[cap_r][col] = 4
                        for cc in range(col + 1, min(hend + 2, cols)):
                            if cc <= hend:
                                if result[cap_r][cc] in [0, 7]:
                                    result[cap_r][cc] = 2
                            elif cc == hend + 1:
                                result[cap_r][cc] = 3
                                # Spawn line from cap edge in SAME direction
                                queue.append((cap_r, cc, direction))

                    # Stop at real intersection
                    break
                else:
                    # Hit something else, stop
                    break
            else:
                # No intersection, draw line segment
                if result[r][col] == 0:
                    # Check if this is a gap in a horizontal line
                    if (r, col) in line_gaps:
                        result[r][col] = 6
                    else:
                        result[r][col] = 2
                elif result[r][col] in [2, 3, 4, 6]:
                    # Hit existing structure
                    break
                else:
                    break
                r += step

    # Post-process: place 6s at top row where vertical lines terminate
    for c in range(cols):
        if result[0][c] == 2:
            result[0][c] = 6

    return result
