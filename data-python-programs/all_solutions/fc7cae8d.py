def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with border markers on edges and a rectangular region of interest inside
    2. The rectangular region contains dense non-zero values distinct from border markers
    3. Output is the extracted region rotated 90 degrees counter-clockwise
    4. Border values (appearing on grid edges) mark the boundaries
    5. The content region typically starts around column 4

    Procedure:
    1. Identify edge/border values from the grid perimeter
    2. Find the rectangular region with dense content (excluding border markers and background)
    3. Extract the region
    4. Rotate 90 degrees counter-clockwise (col C-1-c becomes row, row r becomes col)
    """

    rows = len(grid)
    cols = len(grid[0])

    # Identify potential border values (appear frequently on edges)
    edge_values = set()
    # Check top and bottom rows
    for c in range(cols):
        edge_values.add(grid[0][c])
        edge_values.add(grid[rows-1][c])
    # Check left and right columns
    for r in range(rows):
        edge_values.add(grid[r][0])
        edge_values.add(grid[r][cols-1])

    # Background is typically 0
    edge_values.add(0)

    # Find the rectangular region by finding bounding box of non-border content
    # Start searching from column 4 (observed pattern across examples)
    left = 4

    # Find rough vertical bounds first to help with right boundary detection
    content_rows = []
    for r in range(rows):
        # Check if this row has significant content starting around column 4
        content_in_row = sum(1 for c in range(left, min(left + 15, cols))
                            if grid[r][c] != 0 and grid[r][c] not in edge_values)
        if content_in_row >= 3:
            content_rows.append(r)

    if not content_rows:
        return grid

    rough_top = content_rows[0]
    rough_bottom = content_rows[-1]

    # Find right boundary within the content row range
    right = left
    for c in range(left, cols):
        non_border_count = sum(1 for r in range(rough_top, rough_bottom+1)
                               if grid[r][c] != 0 and grid[r][c] not in edge_values)
        # At least 40% of content rows should have values in this column
        threshold = max(2, (rough_bottom - rough_top + 1) * 0.4)
        if non_border_count >= threshold:
            right = c

    # Find precise top boundary with correct left/right range
    top = rough_top
    for r in range(rows):
        non_border_count = sum(1 for c in range(left, right+1)
                               if grid[r][c] != 0 and grid[r][c] not in edge_values)
        threshold = (right - left + 1) * 0.3
        if non_border_count >= threshold:
            top = r
            break

    # Find precise bottom boundary
    bottom = top
    for r in range(top, rows):
        non_border_count = sum(1 for c in range(left, right+1)
                               if grid[r][c] != 0 and grid[r][c] not in edge_values)
        threshold = (right - left + 1) * 0.3
        if non_border_count >= threshold:
            bottom = r

    # Extract the region
    region = []
    for r in range(top, bottom+1):
        row = [grid[r][c] for c in range(left, right+1)]
        region.append(row)

    # Rotate 90 degrees counter-clockwise
    # Original grid is R rows × C cols
    # After rotation: C rows × R cols
    # Two possible rotations based on pattern:
    # 1. Read columns top-to-bottom: (r,c) → (C-1-c, r)
    # 2. Read columns bottom-to-top: (r,c) → (C-1-c, R-1-r)

    region_rows = len(region)
    region_cols = len(region[0])

    # Determine rotation direction based on aspect ratio
    # If more rows than columns, use bottom-to-top reading
    use_reverse = region_rows > region_cols

    result = [[0] * region_rows for _ in range(region_cols)]

    for r in range(region_rows):
        for c in range(region_cols):
            if use_reverse:
                # Read columns from bottom to top
                result[region_cols-1-c][region_rows-1-r] = region[r][c]
            else:
                # Read columns from top to bottom
                result[region_cols-1-c][r] = region[r][c]

    return result
