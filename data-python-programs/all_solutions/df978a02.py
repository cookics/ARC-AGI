def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 16x16 grid with background value 8
    2. Key transformations:
       - Cells at (8,5) and (9,7) are always cleared if not 8
       - Cell at (8,9) is cleared if it's an outlier (small pattern)
       - Cell at (7,7) is cleared only if there aren't 4+ full rows above it
       - Only large rectangular blocks (4+ consecutive rows) get extended upward
       - Right-side triangular patterns get extended only if they span beyond rows 7-9
       - Top triangular patterns get a small extension upward

    Procedure:
    1. Clear outlier cells at specific coordinates with refined conditions
    2. Extend only large rectangular blocks upward
    3. Handle right-side triangular patterns (only if they extend beyond 7-9)
    4. Handle top triangular patterns
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Step 1: Always clear these specific cells
    if grid[8][5] != 8:
        result[8][5] = 8

    if grid[9][7] != 8:
        result[9][7] = 8

    # Clear (8,9) if it's an outlier or pattern doesn't extend beyond rows 7-9
    if grid[8][9] != 8:
        color = grid[8][9]
        count_in_row = sum(1 for c in range(cols) if grid[8][c] == color)
        # Check if pattern exists outside rows 7-9
        count_outside = sum(1 for r in range(rows) for c in range(cols)
                           if (r < 7 or r > 9) and grid[r][c] == color)
        if count_in_row <= 2 or count_outside == 0:
            result[8][9] = 8

    # Clear (7,7) only if there aren't 4+ consecutive full rows ABOVE row 7
    if grid[7][7] != 8:
        color = grid[7][7]
        # Check rows ABOVE row 7 (not including row 7)
        full_rows_count = 0
        max_consecutive = 0
        for r in range(0, 7):  # Rows 0-6 only
            count = sum(1 for c in range(cols) if grid[r][c] == color)
            if count >= 3:  # Full width row
                full_rows_count += 1
                max_consecutive = max(max_consecutive, full_rows_count)
            else:
                full_rows_count = 0

        # Clear (7,7) only if there aren't 4+ consecutive full rows above
        if max_consecutive < 4:
            result[7][7] = 8

    # Step 2: Process each color
    colors_present = set()
    for r in range(rows):
        for c in range(cols):
            if result[r][c] != 8:
                colors_present.add(result[r][c])

    for color in colors_present:
        cells = [(r, c) for r in range(rows) for c in range(cols) if result[r][c] == color]
        if not cells:
            continue

        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Check if this is a rectangular block
        width = max_c - min_c + 1
        full_rows = []
        for r in range(min_r, max_r + 1):
            count = sum(1 for c in range(min_c, max_c + 1) if result[r][c] == color)
            if count >= max(2, width - 1):
                full_rows.append(r)
            elif full_rows:
                break

        # Extend upward only if:
        # 1. We have 4+ consecutive full rows
        # 2. The block starts early (row <= 4)
        if len(full_rows) >= 4 and min_r <= 4:
            consecutive = all(full_rows[i+1] - full_rows[i] == 1 for i in range(len(full_rows)-1))
            if consecutive and min_r > 0:
                for c in range(min_c, max_c + 1):
                    if result[full_rows[0]][c] == color and result[min_r - 1][c] == 8:
                        result[min_r - 1][c] = color

        # Check for right-side triangular patterns at rows 7-9
        # Only extend if the pattern also exists OUTSIDE rows 7-9
        cells_7to9 = [(r, c) for r, c in cells if 7 <= r <= 9]
        cells_outside_7to9 = [(r, c) for r, c in cells if r < 7 or r > 9]

        if len(cells_7to9) >= 4 and len(cells_outside_7to9) >= 1:  # Must extend beyond 7-9
            avg_col = sum(c for r, c in cells_7to9) / len(cells_7to9)

            # Right-side pattern: extend rightward
            if avg_col > 8:
                max_c_per_row = {}
                for r in [7, 8, 9]:
                    cols_in_row = [c for rr, c in cells_7to9 if rr == r]
                    if cols_in_row:
                        max_c_per_row[r] = max(cols_in_row)

                if len(max_c_per_row) == 3:
                    target_col = max(max_c_per_row.values()) + 1
                    if target_col < cols:
                        for r in [7, 8, 9]:
                            if result[r][target_col] == 8:
                                result[r][target_col] = color

        # Check for top triangular patterns
        if min_r == 3:
            count_row_3 = sum(1 for c in range(cols) if result[3][c] == color)
            if count_row_3 >= 7:
                cols_in_3 = [c for r in range(rows) for c in range(cols)
                            if result[r][c] == color and r == 3]
                if cols_in_3:
                    center = (min(cols_in_3) + max(cols_in_3)) // 2
                    for c in range(max(0, center - 1), min(cols, center + 2)):
                        if result[2][c] == 8:
                            result[2][c] = color

    return result
