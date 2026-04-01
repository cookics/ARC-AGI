def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x15 grid with three sections:
       - Top section (rows 0-2): contains 2s and 0s pattern
       - Middle section (rows 3-5): all 0s (separator)
       - Bottom section (rows 6-9): contains 5s and 0s pattern
    2. Output transformation:
       - Rows 0-2: 0s become 1s, 2s stay as 2s
       - Rows 5-9: all become 0s
       - Rows 3-4: complex projection from bottom section, filtered by top pattern
    3. The 5s in bottom section "extend upward" to rows 3-4 based on their positions
       - Column characteristics determine which rows get filled with 1s

    Procedure:
    1. Convert all 0s to 1s in top section (rows 0-2)
    2. Determine filling pattern for rows 3-4 based on bottom section 5s
    3. Clear rows 5-9
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Step 1: In top section (rows 0-2), convert all 0s to 1s
    for r in range(3):
        for c in range(cols):
            if result[r][c] == 0:
                result[r][c] = 1

    # Step 2: Clear all rows from 3 onwards
    for r in range(3, rows):
        for c in range(cols):
            result[r][c] = 0

    # Step 3: Fill rows 3 and 4 based on bottom section pattern
    # First, identify connected components and their properties
    def get_component_info(c):
        """Get info about the connected component containing column c"""
        if not any(grid[r][c] == 5 for r in range(6, 10)):
            return None

        # Find the extent of this component
        left = c
        while left > 0 and any(grid[r][left-1] == 5 for r in range(6, 10)):
            left -= 1
        right = c
        while right < cols - 1 and any(grid[r][right+1] == 5 for r in range(6, 10)):
            right += 1

        # Check if component has 5s in early rows (6-7)
        has_early_5s = any(grid[r][c2] == 5 for r in range(6, 8) for c2 in range(left, right+1))

        return {'left': left, 'right': right, 'has_early_5s': has_early_5s}

    for c in range(cols):
        has_zero_in_top = (grid[1][c] == 0 or grid[2][c] == 0)

        # Find topmost 5 in this column
        topmost_5 = None
        for r in range(6, 10):
            if grid[r][c] == 5:
                topmost_5 = r
                break

        comp_info = get_component_info(c)
        is_in_component = comp_info is not None
        is_left_edge = is_in_component and c == comp_info['left']
        is_right_edge = is_in_component and c == comp_info['right']
        is_single_col = is_in_component and comp_info['left'] == comp_info['right']
        has_early_5s = is_in_component and comp_info['has_early_5s']

        # Check if left edge of component has 0 in top
        left_edge_has_zero = False
        if is_in_component:
            left_col = comp_info['left']
            left_edge_has_zero = (grid[1][left_col] == 0 or grid[2][left_col] == 0)

        # Check if right neighbor has component info
        right_comp = get_component_info(c+1) if c < cols - 1 else None
        right_is_left_edge = right_comp is not None and c+1 == right_comp['left']
        right_has_zero = (c < cols - 1) and (grid[1][c+1] == 0 or grid[2][c+1] == 0)

        # Check if column is in a single-column gap between components
        is_single_gap = (not is_in_component and has_zero_in_top and
                         c > 0 and c < cols - 1 and
                         any(grid[r][c-1] == 5 for r in range(6, 10)) and
                         any(grid[r][c+1] == 5 for r in range(6, 10)))

        # Single gap adjacent to left edge with 0 should fill
        single_gap_fills = is_single_gap and right_is_left_edge and right_has_zero

        # Row 3 logic
        if has_zero_in_top:
            if is_single_gap and not single_gap_fills:
                pass  # Don't fill most single gaps
            elif single_gap_fills:
                result[3][c] = 1  # Fill special single gaps
            elif is_in_component and (has_early_5s or left_edge_has_zero):
                result[3][c] = 1
            elif not is_in_component:
                result[3][c] = 1
        elif is_in_component:
            # No 0 in top but has 5s
            if has_early_5s and not is_right_edge:
                result[3][c] = 1
            elif not has_early_5s and left_edge_has_zero:
                # Fill all except rightmost, unless rightmost is not at boundary
                if not is_right_edge:
                    result[3][c] = 1
                elif c < cols - 1 and not any(grid[r][c+1] == 5 for r in range(6, 10)):
                    result[3][c] = 1
            elif is_single_col and topmost_5 >= 7:
                # Single column with topmost at row 7+
                result[3][c] = 1

        # Row 4 logic
        if has_zero_in_top:
            if is_left_edge and topmost_5 == 9:
                result[4][c] = 1
            elif single_gap_fills:
                result[4][c] = 1
            elif not is_in_component and c == cols - 1:
                # Last column with 0 in top and no 5s
                result[4][c] = 1

    return result
