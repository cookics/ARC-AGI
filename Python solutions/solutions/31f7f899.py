def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a special "anchor" row filled mostly with 6s
    2. Vertical lines of non-background values extend above/below this anchor row
    3. Each vertical line has an "extent" (number of rows above/below anchor)
    4. Output: the extents are sorted in ascending order and reassigned left-to-right
    5. Background value is 8
    6. Example: If vertical lines at columns [A, B, C] have extents [3, 1, 2],
       output will have extents [1, 2, 3] at columns [A, B, C]

    Procedure:
    1. Find the anchor row (the row with most 6s)
    2. Identify vertical line columns (non-6 values in anchor row)
    3. For each line, calculate its extent (symmetric distance from anchor)
    4. Sort the extents in ascending order
    5. Assign sorted extents back to lines from left to right
    6. Reconstruct the grid with new extents
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find anchor row (row with the most 6s)
    anchor_row = -1
    max_sixes = 0
    for r in range(rows):
        six_count = sum(1 for c in range(cols) if grid[r][c] == 6)
        if six_count > max_sixes:
            max_sixes = six_count
            anchor_row = r

    # Identify vertical lines in anchor row
    # Group consecutive columns with same non-6 value
    lines = []  # List of (columns, value) tuples
    c = 0
    while c < cols:
        if grid[anchor_row][c] != 6:
            value = grid[anchor_row][c]
            start_col = c
            # Find consecutive columns with same value
            while c < cols and grid[anchor_row][c] == value:
                c += 1
            line_cols = list(range(start_col, c))
            lines.append((line_cols, value))
        else:
            c += 1

    # Calculate extent for each line
    extents = []
    for line_cols, value in lines:
        # Check how far the line extends above and below anchor
        # Use first column of the line group
        col = line_cols[0]

        # Count rows above anchor
        above = 0
        for r in range(anchor_row - 1, -1, -1):
            if grid[r][col] == value:
                above += 1
            else:
                break

        # Count rows below anchor
        below = 0
        for r in range(anchor_row + 1, rows):
            if grid[r][col] == value:
                below += 1
            else:
                break

        # Extent is the minimum of above and below (for symmetry)
        extent = min(above, below)
        extents.append(extent)

    # Sort extents
    sorted_extents = sorted(extents)

    # Create output grid (copy of input initially)
    result = [row[:] for row in grid]

    # Clear old vertical lines (except anchor row)
    for r in range(rows):
        if r != anchor_row:
            for c in range(cols):
                if result[r][c] != 6 and result[r][c] != 8:
                    result[r][c] = 8

    # Draw new vertical lines with sorted extents
    for i, (line_cols, value) in enumerate(lines):
        new_extent = sorted_extents[i]

        # Draw above anchor
        for r in range(anchor_row - new_extent, anchor_row):
            for col in line_cols:
                result[r][col] = value

        # Draw below anchor
        for r in range(anchor_row + 1, anchor_row + new_extent + 1):
            for col in line_cols:
                result[r][col] = value

    return result
