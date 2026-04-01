def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has three regions: top (1s), middle (pattern with 5s), bottom (6s)
    2. Pattern with 5s appears on either left or right side
    3. Pattern has solid 5-rows followed by alternating 5-1 rows
    4. Output moves pattern to opposite horizontal side
    5. Solid rows move intact, alternating rows shift diagonally away from origin
    6. Boundary row (first 6-row) gets filled with 9s in diagonal extension area

    Procedure:
    1. Find the 5-pattern block boundaries
    2. Determine if it's on left or right side
    3. Move to opposite side with diagonal shifting for alternating rows
    4. Fill boundary row with 9s
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find the 5-pattern boundaries
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    assert min_row != rows, "No 5-pattern found"

    pattern_width = max_col - min_col + 1
    pattern_height = max_row - min_row + 1

    # Determine if pattern is on left or right side
    on_left_side = min_col < cols // 2

    # Calculate target position (opposite side)
    if on_left_side:
        target_col = cols - pattern_width
    else:
        target_col = 0

    # Clear original pattern area
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if grid[r][c] == 5:
                result[r][c] = 1

    # Find solid vs alternating rows
    solid_rows = []
    alternating_rows = []

    for r in range(min_row, max_row + 1):
        row_has_alternating = False
        for c in range(min_col, max_col + 1):
            if grid[r][c] == 1:  # Has 1s mixed with 5s
                row_has_alternating = True
                break

        if row_has_alternating:
            alternating_rows.append(r)
        else:
            solid_rows.append(r)

    # Move solid rows as-is
    for r in solid_rows:
        for c in range(pattern_width):
            if grid[r][min_col + c] == 5:
                result[r][target_col + c] = 5

    # Move alternating rows with diagonal shift
    for i, r in enumerate(alternating_rows):
        if on_left_side:
            # Moving to right, shift left (decreasing columns)
            shift = i
            actual_target = target_col - shift
        else:
            # Moving to left, shift right (increasing columns)
            shift = i
            actual_target = target_col + shift

        # Copy the pattern with shift
        for c in range(pattern_width):
            src_col = min_col + c
            dst_col = actual_target + c

            if 0 <= dst_col < cols and grid[r][src_col] == 5:
                result[r][dst_col] = 5

    # Find boundary row (first row with 6s)
    boundary_row = -1
    for r in range(rows):
        if grid[r][0] == 6:
            boundary_row = r
            break

    assert boundary_row != -1, "No boundary row found"

    # Fill boundary row with 9s in appropriate region
    if on_left_side:
        # Pattern moved from left to right, 9s fill up to target_col + 1
        fill_end = target_col + 2
        for c in range(fill_end):
            if result[boundary_row][c] == 6:
                result[boundary_row][c] = 9
    else:
        # Pattern moved from right to left, 9s start after the target pattern
        fill_start = target_col + pattern_width
        for c in range(fill_start, cols):
            if result[boundary_row][c] == 6:
                result[boundary_row][c] = 9

    return result
