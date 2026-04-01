def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has border columns/rows with uniform values at edges
    2. Interior divided by separator columns or rows
    3. Sections are swapped and each rotated 180 degrees
    4. Vertical separator: LEFT ↔ RIGHT sections (swap + rotate 180°)
    5. Horizontal separator: TOP ↔ BOTTOM sections (swap + rotate 180°)

    Procedure:
    1. Detect borders
    2. Detect separators
    3. Swap and rotate sections
    """

    from collections import Counter

    H, W = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Helper: rotate 180°
    def rot180(sec):
        return [row[::-1] for row in sec[::-1]]

    # Find borders (uniform edge columns/rows)
    left_bd = 0
    while left_bd < W and all(grid[r][left_bd] == grid[0][left_bd] for r in range(H)):
        left_bd += 1

    right_bd = W
    while right_bd > left_bd and all(grid[r][right_bd-1] == grid[0][right_bd-1] for r in range(H)):
        right_bd -= 1

    top_bd = 0
    while top_bd < H and len(set(grid[top_bd])) == 1:
        top_bd += 1

    bot_bd = H
    while bot_bd > top_bd and len(set(grid[bot_bd-1])) == 1:
        bot_bd -= 1

    # Find vertical separator (uniform column in interior)
    v_sep = None
    for c in range(left_bd, right_bd):
        col = [grid[r][c] for r in range(top_bd, bot_bd)]
        if len(set(col)) == 1:
            v_sep = c
            break

    # Find horizontal separator (uniform row in interior)
    h_sep = None
    for r in range(top_bd, bot_bd):
        row = [grid[r][c] for c in range(left_bd, right_bd)]
        if len(set(row)) <= 2:
            cnt = Counter(row)
            if len(cnt) > 0 and cnt.most_common(1)[0][1] >= len(row) * 0.9:
                h_sep = r
                break

    # Transform based on separator type
    if v_sep is not None:
        # Vertical separator: swap left and right with 180° rotation
        # Expand to include consecutive uniform columns
        v_start, v_end = v_sep, v_sep
        while v_start > left_bd and all(grid[r][v_start-1] == grid[0][v_start-1] for r in range(H)):
            v_start -= 1
        while v_end < right_bd - 1 and all(grid[r][v_end+1] == grid[0][v_end+1] for r in range(H)):
            v_end += 1

        # Extract left and right sections
        left_sec = [[grid[r][c] for c in range(left_bd, v_start)] for r in range(top_bd, bot_bd)]
        right_sec = [[grid[r][c] for c in range(v_end+1, right_bd)] for r in range(top_bd, bot_bd)]

        # Rotate and swap
        left_rot = rot180(left_sec)
        right_rot = rot180(right_sec)

        # Place rotated right in left position
        for i in range(len(right_rot)):
            for j in range(len(right_rot[i])):
                result[top_bd + i][left_bd + j] = right_rot[i][j]

        # Place rotated left in right position
        for i in range(len(left_rot)):
            for j in range(len(left_rot[i])):
                result[top_bd + i][v_end + 1 + j] = left_rot[i][j]

    elif h_sep is not None:
        # Horizontal separator: swap top and bottom with 180° rotation
        # Expand to include consecutive uniform rows
        h_start, h_end = h_sep, h_sep
        while h_start > top_bd:
            row = [grid[h_start-1][c] for c in range(left_bd, right_bd)]
            if len(set(row)) <= 2 and Counter(row).most_common(1)[0][1] >= len(row) * 0.9:
                h_start -= 1
            else:
                break
        while h_end < bot_bd - 1:
            row = [grid[h_end+1][c] for c in range(left_bd, right_bd)]
            if len(set(row)) <= 2 and Counter(row).most_common(1)[0][1] >= len(row) * 0.9:
                h_end += 1
            else:
                break

        # Extract top and bottom sections
        top_sec = [[grid[r][c] for c in range(left_bd, right_bd)] for r in range(top_bd, h_start)]
        bot_sec = [[grid[r][c] for c in range(left_bd, right_bd)] for r in range(h_end+1, bot_bd)]

        # Rotate and swap
        top_rot = rot180(top_sec)
        bot_rot = rot180(bot_sec)

        # Place rotated bottom in top position
        for i in range(len(bot_rot)):
            for j in range(len(bot_rot[i])):
                result[top_bd + i][left_bd + j] = bot_rot[i][j]

        # Place rotated top in bottom position
        for i in range(len(top_rot)):
            for j in range(len(top_rot[i])):
                result[h_end + 1 + i][left_bd + j] = top_rot[i][j]

    return result
