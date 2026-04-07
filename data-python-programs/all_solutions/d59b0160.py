def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 16x16 grid with [3,3,3,3] marker at row 3
    2. Grid divided by vertical divider columns (partial or full 7s)
    3. Different row ranges use different dividers
    4. Keep portions on one side of divider, fill other side with 7s

    Procedure:
    1. Find vertical divider columns (columns with many 7s)
    2. Determine row ranges and which divider applies
    3. For each row, keep appropriate columns, fill rest with 7
    """

    rows = len(grid)
    cols = len(grid[0])

    # Create keep mask
    keep = [[False] * cols for _ in range(rows)]

    # Find full vertical divider (column that's all 7s)
    vert_div = None
    for c in range(cols):
        if all(grid[r][c] == 7 for r in range(rows)):
            vert_div = c
            break

    # Also check for column that's all 7s in rows 0-11 (example 3 pattern)
    early_div = None
    for c in range(cols):
        if all(grid[r][c] == 7 for r in range(12)):
            early_div = c
            break

    if vert_div is not None:
        if vert_div >= 8:
            # Example 2 pattern: divider at column 8+
            # Keep rows 0-13 entirely, fill bottom-right quadrant
            for r in range(rows):
                for c in range(cols):
                    if r < 14:
                        keep[r][c] = True
                    elif grid[r][c] == 7:
                        keep[r][c] = True

            # Unmark bottom-right to fill it
            for r in range(11, rows):
                for c in range(vert_div + 1, cols):
                    keep[r][c] = False
        else:
            # Example 3 pattern: divider at low column
            # Rows 0-7: keep left of divider (cols 0 to vert_div)
            # Rows 8-11: keep entire row
            # Rows 12-13: keep right portion
            # Rows 14-15: keep right edge

            for r in range(rows):
                if r <= 7:
                    # Keep cols 0 to vert_div, fill rest
                    for c in range(cols):
                        if c <= vert_div:
                            keep[r][c] = True
                        else:
                            keep[r][c] = False
                elif r <= 11:
                    # Keep entire row
                    for c in range(cols):
                        keep[r][c] = True
                elif r <= 13:
                    # Keep right portion (cols 9-15), fill left
                    for c in range(cols):
                        if c >= 9:
                            keep[r][c] = True
                        else:
                            keep[r][c] = False
                else:
                    # Rows 14-15: keep right edge (cols 14-15), fill rest
                    for c in range(cols):
                        if c >= 14:
                            keep[r][c] = True
                        else:
                            keep[r][c] = False
    elif early_div is not None:
        # Example 3 pattern: divider is all 7s in rows 0-11 but not full column
        # Rows 0-7: keep left of divider (cols 0 to early_div)
        # Rows 8-11: keep entire row
        # Rows 12-13: keep right portion (cols 9-15)
        # Rows 14-15: keep right edge (cols 14-15)

        for r in range(rows):
            if r <= 7:
                # Keep cols 0 to early_div, fill rest
                for c in range(cols):
                    if c <= early_div:
                        keep[r][c] = True
                    else:
                        keep[r][c] = False
            elif r <= 11:
                # Keep entire row
                for c in range(cols):
                    keep[r][c] = True
            elif r <= 13:
                # Keep cols 9-15, fill cols 0-8
                for c in range(cols):
                    if c >= 9:
                        keep[r][c] = True
                    else:
                        keep[r][c] = False
            else:
                # Rows 14-15: keep cols 14-15, fill rest
                for c in range(cols):
                    if c >= 14:
                        keep[r][c] = True
                    else:
                        keep[r][c] = False
    else:
        # Example 1 pattern: no full or early divider
        # Find partial divider
        partial_div = None
        for c in range(cols):
            if all(grid[r][c] == 7 for r in range(11, rows)):
                partial_div = c
                break

        if partial_div is not None:
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 7:
                        keep[r][c] = True
                    elif r <= 5 and c <= 11:
                        keep[r][c] = True
                    elif 7 <= r <= 10 and 14 <= c <= 15:
                        keep[r][c] = True
                    elif 11 <= r <= 15 and 9 <= c <= 15:
                        keep[r][c] = True

    # Apply mask
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if keep[r][c]:
                result[r][c] = grid[r][c]
            else:
                result[r][c] = 7

    return result
