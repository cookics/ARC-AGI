def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains exactly 2 cells with value 3 (adjacent horizontally or vertically)
    2. Input contains values 7 and 8, with 8s acting as obstacles
    3. Output fills certain 7s with 3s in a cross-like pattern
    4. The pattern has a "focus" point that determines the cross center
    5. For horizontal 3s: focus moves toward grid center/opposite side
    6. For vertical 3s: focus is at the top 3's position
    7. Cross pattern extends with alternating fill density based on distance

    Procedure:
    1. Find the two 3s and determine orientation (horizontal/vertical)
    2. Find reference row (row with most 7s, excluding rows with 3s)
    3. Determine focus point based on orientation and position
    4. Fill cross pattern from focus:
       - Horizontal line at focus row (all 7s)
       - Vertical line at focus column (up to bottom 3)
       - Rows above focus: even distance = fill all, odd distance = fill odd columns
       - Rows below focus at/past 3s: even distance = fill all, odd = fill odd columns
    """

    import copy
    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find input 3s
    threes = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                threes.append((r, c))

    if len(threes) != 2:
        return result

    min_row_3 = min(t[0] for t in threes)
    max_row_3 = max(t[0] for t in threes)
    min_col_3 = min(t[1] for t in threes)
    max_col_3 = max(t[1] for t in threes)

    # Determine orientation
    is_vertical = (max_row_3 > min_row_3)

    # Find row with most 7s (excluding rows with 3s)
    best_row = -1
    max_sevens = -1
    for r in range(rows):
        if min_row_3 <= r <= max_row_3:
            continue
        count_sevens = sum(1 for c in range(cols) if grid[r][c] == 7)
        if count_sevens > max_sevens:
            max_sevens = count_sevens
            best_row = r

    # Determine focus point
    midpoint_row = rows // 2
    midpoint_col = cols // 2

    if is_vertical:
        # Vertical 3s: focus at top of 3s
        focus_col = min_col_3
        focus_row = min_row_3
    else:
        # Horizontal 3s: determine focus based on position
        if min_row_3 < midpoint_row:
            # 3s in top half: focus below
            focus_row = min_row_3 + 2
        else:
            # 3s in bottom half: focus above
            if best_row >= 0 and best_row < min_row_3:
                focus_row = best_row + 1
            else:
                focus_row = min_row_3 - 2

        # Determine focus column based on 3s position
        if max_col_3 < midpoint_col:
            # 3s on left: focus to the right
            focus_col = max_col_3 + 2
        else:
            # 3s on right: focus to the left
            focus_col = min_col_3 - 2

    # Ensure focus is in bounds
    focus_row = max(0, min(focus_row, rows - 1))
    focus_col = max(0, min(focus_col, cols - 1))

    # Fill pattern
    # a. Horizontal cross (focus row) - fill all 7s
    for c in range(cols):
        if result[focus_row][c] == 7:
            result[focus_row][c] = 3

    # b. Vertical cross (focus column) - fill up to max_row_3
    for r in range(max_row_3 + 1):
        if result[r][focus_col] == 7:
            result[r][focus_col] = 3

    # c. Rows above the focus
    for r in range(focus_row):
        if grid[r][focus_col] == 8:
            continue  # Skip rows where focus column has 8
        distance = focus_row - r
        if distance % 2 == 0:  # even distance
            # Fill all 7s from focus_col to right
            for c in range(focus_col, cols):
                if result[r][c] == 7:
                    result[r][c] = 3
        else:  # odd distance
            # Fill 7s at odd-indexed columns from focus_col to right
            for c in range(focus_col, cols):
                if c % 2 == 1 and result[r][c] == 7:
                    result[r][c] = 3

    # d. Rows at input 3s (only between min and max, not beyond)
    for r in range(min_row_3, max_row_3 + 1):
        if r == focus_row:
            continue  # Already handled
        distance = r - focus_row
        if distance % 2 == 0:  # even distance
            # Fill all 7s from left to focus_col
            for c in range(focus_col + 1):
                if result[r][c] == 7:
                    result[r][c] = 3
        else:  # odd distance
            # Fill 7s at odd-indexed columns from left to focus_col
            for c in range(focus_col + 1):
                if c % 2 == 1 and result[r][c] == 7:
                    result[r][c] = 3

    return result
