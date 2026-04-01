def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has one anchor (value 3) and segments containing both 1s and 2s
    2. All 1s and 2s in segments are replaced with 2
    3. Frames made of 3s connect anchor to segments creating L-shaped partitions
    4. Horizontal segments get vertical partitions, vertical segments get horizontal frames

    Procedure:
    1. Find anchor (3) position
    2. Find all segments (horizontal/vertical sequences with both 1 and 2)
    3. Replace segment values with 2
    4. Draw L-shaped frames connecting anchor to each segment
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find anchor (3)
    r3, c3 = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                r3, c3 = r, c
                break
        if r3 is not None:
            break

    # Find horizontal segments (rows with both 1 and 2)
    h_segs = []
    for r in range(rows):
        has_1, has_2 = False, False
        c_min, c_max = None, None
        for c in range(cols):
            if grid[r][c] in [1, 2]:
                if c_min is None:
                    c_min = c
                c_max = c
                if grid[r][c] == 1:
                    has_1 = True
                elif grid[r][c] == 2:
                    has_2 = True
        if has_1 and has_2:
            h_segs.append((r, c_min, c_max))

    # Find vertical segments (continuous columns with both 1 and 2)
    v_segs = []
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r][c] in [1, 2]:
                start = r
                has_1, has_2 = False, False
                while r < rows and grid[r][c] in [1, 2]:
                    if grid[r][c] == 1:
                        has_1 = True
                    elif grid[r][c] == 2:
                        has_2 = True
                    r += 1
                if has_1 and has_2:
                    v_segs.append((c, start, r - 1))
            else:
                r += 1

    # Replace all segment values with 2
    for r, c_min, c_max in h_segs:
        for c in range(c_min, c_max + 1):
            result[r][c] = 2

    for c, r_min, r_max in v_segs:
        for r in range(r_min, r_max + 1):
            result[r][c] = 2

    # Draw frames for horizontal segments
    if h_segs:
        prev_r = r3
        prev_c = c3

        for idx, (seg_r, c_min, c_max) in enumerate(h_segs):
            # Determine turn row (partition point)
            if idx == 0:
                turn_r = r3 + 2
            else:
                prev_seg_r = h_segs[idx - 1][0]
                turn_r = (prev_seg_r + seg_r) // 2

            # Determine partition column
            if len(h_segs) > 1 and idx == 0:
                part_c = c3 + 4
            else:
                part_c = 1

            # Draw vertical line from previous position to turn row
            for r in range(prev_r, turn_r + 1):
                if result[r][prev_c] != 2:
                    result[r][prev_c] = 3

            # Draw horizontal line at turn row
            c_start, c_end = min(prev_c, part_c), max(prev_c, part_c)
            for c in range(max(0, c_start), min(cols, c_end + 1)):
                if result[turn_r][c] != 2:
                    result[turn_r][c] = 3

            # Draw vertical line from turn row to bottom
            if 0 <= part_c < cols:
                for r in range(turn_r, rows):
                    if result[r][part_c] != 2:
                        result[r][part_c] = 3

            prev_r = turn_r
            prev_c = part_c

    # Draw frames for vertical segments
    if v_segs and len(v_segs) >= 2:
        # Create rectangular frame between vertical segments
        first_c, first_r_min, first_r_max = v_segs[0]
        last_c, last_r_min, last_r_max = v_segs[-1]

        # Frame dimensions
        top_r = r3
        bot_r = max(first_r_max + 2, r3 + 4)
        left_c = first_c - r3 + 1
        right_c = last_c - first_r_max + first_r_min - 1 + r3

        # Top horizontal line - left segment (from anchor to left_c)
        for c in range(max(0, c3), min(cols, left_c + 1)):
            if result[top_r][c] != 2:
                result[top_r][c] = 3

        # Top horizontal line - right segment (from right_c to right edge)
        for c in range(max(0, right_c), cols):
            if result[top_r][c] != 2:
                result[top_r][c] = 3

        # Left and right vertical lines
        for r in range(top_r + 1, min(bot_r + 1, rows)):
            if 0 <= left_c < cols and result[r][left_c] != 2:
                result[r][left_c] = 3
            if 0 <= right_c < cols and result[r][right_c] != 2:
                result[r][right_c] = 3

        # Bottom horizontal line
        if bot_r < rows:
            for c in range(max(0, left_c), min(cols, right_c + 1)):
                if result[bot_r][c] != 2:
                    result[bot_r][c] = 3

    return result
