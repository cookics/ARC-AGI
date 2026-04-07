def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains two distinct colored patterns: 2s and 5s on a background of 7s
    2. Output shows these patterns moved toward each other and made adjacent
    3. The patterns become mirror images of each other across the boundary between them
    4. The 2s always move by 2 units toward the 5s
    5. The 5s are repositioned to be the mirror reflection of the 2s

    Procedure:
    1. Find all positions of 2s and 5s
    2. Determine if separation is primarily horizontal or vertical
    3. Move 2s by 2 units toward 5s
    4. Calculate the mirror axis (boundary between the patterns)
    5. Mirror the 2s across this axis to determine 5s positions
    6. Place both patterns in the output grid
    """

    rows, cols = len(grid), len(grid[0])

    # Find all 2s and 5s
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    fives = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]

    if not twos or not fives:
        return grid

    # Calculate centers to determine primary separation direction
    r2_center = sum(r for r, _ in twos) / len(twos)
    c2_center = sum(c for _, c in twos) / len(twos)
    r5_center = sum(r for r, _ in fives) / len(fives)
    c5_center = sum(c for _, c in fives) / len(fives)

    h_dist = abs(c2_center - c5_center)
    v_dist = abs(r2_center - r5_center)

    # Initialize result grid
    result = [row[:] for row in grid]

    # Clear old positions
    for r, c in twos + fives:
        result[r][c] = 7

    if h_dist > v_dist:
        # Horizontal separation: move 2s by 2 columns toward 5s
        shift_c = 2 if c2_center < c5_center else -2

        # Place 2s at new positions
        for r, c in twos:
            new_c = c + shift_c
            result[r][new_c] = 2

        # Calculate mirror axis (boundary between patterns)
        c2_max_new = max(c for _, c in twos) + shift_c
        axis = c2_max_new + 0.5

        # Mirror 2s to create 5s
        for r, c in twos:
            new_c = c + shift_c
            mirror_c = int(2 * axis - new_c)
            result[r][mirror_c] = 5
    else:
        # Vertical separation: move 2s by 2 rows toward 5s
        if r2_center < r5_center:
            # 2s above 5s: move down
            shift_r = 2
            r2_max_new = max(r for r, _ in twos) + shift_r
            axis = r2_max_new + 0.5
        else:
            # 2s below 5s: move up
            shift_r = -2
            r2_min_new = min(r for r, _ in twos) + shift_r
            axis = r2_min_new - 0.5

        # Place 2s at new positions
        for r, c in twos:
            new_r = r + shift_r
            result[new_r][c] = 2

        # Mirror 2s to create 5s
        for r, c in twos:
            new_r = r + shift_r
            mirror_r = int(2 * axis - new_r)
            result[mirror_r][c] = 5

    return result
