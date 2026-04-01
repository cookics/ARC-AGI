def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a vertical divider column (all cells have the same non-zero value)
    2. Left region contains patterns made of various non-zero values
    3. Right region has a rectangular frame with an interior
    4. The marker column (immediately after divider) indicates which pattern to place at each row
    5. The top frame row shows where each value should be centered horizontally
    6. Output clears the left region and places patterns in the right interior according to markers

    Procedure:
    1. Find the vertical divider column (all same non-zero value)
    2. Extract patterns from left side for each unique value (as relative coordinates from centroid)
    3. Find horizontal anchor positions for each value from the top frame
    4. Clear the left side and interior
    5. For each row, check marker column value and place corresponding pattern at anchor position
    """
    n_rows = len(grid)
    n_cols = len(grid[0])

    # Find vertical divider column (column with all same non-zero value)
    divider_col = -1
    for col in range(n_cols):
        vals = {grid[row][col] for row in range(n_rows)}
        if len(vals) == 1 and list(vals)[0] != 0:
            divider_col = col
            break

    # Extract patterns from left side by value
    patterns = {}
    for val in range(1, 10):
        coords = [(r, c) for r in range(n_rows) for c in range(divider_col)
                  if grid[r][c] == val]
        if coords:
            # Store pattern as offsets from centroid
            center_row = sum(r for r, c in coords) / len(coords)
            center_col = sum(c for r, c in coords) / len(coords)
            patterns[val] = [(r - center_row, c - center_col) for r, c in coords]

    # Find horizontal anchor columns from top frame (can be multiple per value)
    marker_col = divider_col + 1
    anchors = {}
    for val in patterns:
        anchors[val] = []
        for c in range(marker_col, n_cols):
            if grid[0][c] == val:
                anchors[val].append(c)

    # Build output grid
    result = [row[:] for row in grid]

    # Find which values are actually used in the marker column (interior rows only)
    used_values = set()
    for r in range(1, n_rows - 1):
        val = grid[r][marker_col]
        if val in patterns:
            used_values.add(val)

    # Clear only used patterns from left side
    for r in range(n_rows):
        for c in range(divider_col):
            if grid[r][c] in used_values:
                result[r][c] = 0

    # Clear interior (between frame boundaries)
    for r in range(1, n_rows - 1):
        for c in range(marker_col + 1, n_cols - 1):
            result[r][c] = 0

    # Place patterns according to marker column
    for r in range(1, n_rows - 1):
        val = grid[r][marker_col]
        if val in patterns and val in anchors and anchors[val]:
            # Place pattern centered at each anchor column
            for anchor_col in anchors[val]:
                for dr, dc in patterns[val]:
                    nr = int(round(r + dr))
                    nc = int(round(anchor_col + dc))
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and result[nr][nc] == 0:
                        result[nr][nc] = val

    return result
