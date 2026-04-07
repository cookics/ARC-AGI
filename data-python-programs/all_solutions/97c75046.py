def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with mostly 7s (background), some 0s (forming a region), and exactly one 5 (marker)
    2. Output is the same grid but with the 5 moved to a specific boundary position adjacent to the 0-region
    3. The 0s form different geometric patterns: triangles, diamonds, or irregular shapes with concave corners
    4. The 5 moves to mark specific features: apex of triangle, bottom of diamond, or concave corner

    Procedure:
    1. Find the position of the 5 and all 0 positions
    2. Analyze the geometric pattern of 0s (count per row, leftmost position per row)
    3. Detect pattern type: concave corner, expanding triangle, diamond, or other
    4. Calculate target position based on pattern type
    5. Move 5 to target position (replace old position with 7, new position with 5)
    """

    # Find the position of 5
    r5, c5 = None, None
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 5:
                r5, c5 = r, c
                break
        if r5 is not None:
            break

    # Find all 0 positions
    zeros = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0:
                zeros.append((r, c))

    # Find bounding box of 0s
    rows = [r for r, c in zeros]
    cols = [c for r, c in zeros]
    r0_min, r0_max = min(rows), max(rows)
    c0_min, c0_max = min(cols), max(cols)

    # Count 0s per row
    zeros_per_row = {}
    for r, c in zeros:
        zeros_per_row[r] = zeros_per_row.get(r, 0) + 1

    rows_with_zeros = sorted(zeros_per_row.keys())
    counts = [zeros_per_row[r] for r in rows_with_zeros]

    # Find leftmost 0 per row (for concave corner detection)
    leftmost_per_row = {}
    for r, c in zeros:
        if r not in leftmost_per_row or c < leftmost_per_row[r]:
            leftmost_per_row[r] = c

    # Check for concave corner (leftmost 0 jumps to the right)
    concave_corner = None
    for i in range(1, len(rows_with_zeros)):
        if leftmost_per_row[rows_with_zeros[i]] > leftmost_per_row[rows_with_zeros[i-1]]:
            concave_corner = (rows_with_zeros[i], leftmost_per_row[rows_with_zeros[i]])
            break

    if concave_corner:
        # Concave corner detected - place 5 at second row with 0s, at the concave column
        concave_row, concave_col = concave_corner
        if len(rows_with_zeros) >= 2:
            r5_new = rows_with_zeros[1]  # Second row with 0s
            c5_new = concave_col
        else:
            r5_new, c5_new = r0_min + 1, (c0_min + c0_max) // 2
    elif counts == sorted(counts):  # Increasing - triangle expanding downward
        # Find topmost row with smallest count (apex)
        for r in rows_with_zeros:
            row_zeros = [c for rr, c in zeros if rr == r]
            if len(row_zeros) == min(counts):
                # This is the apex - place 5 one row above
                # Use the column of the apex (rightmost if multiple)
                r5_new, c5_new = r - 1, max(row_zeros)
                break
    else:
        # Diamond or other pattern - use bottom apex strategy
        last_row = rows_with_zeros[-1]
        row_zeros = [c for rr, c in zeros if rr == last_row]
        r5_new, c5_new = last_row + 1, min(row_zeros)

    # Create result
    result = [row[:] for row in grid]
    result[r5][c5] = 7
    result[r5_new][c5_new] = 5
    return result
