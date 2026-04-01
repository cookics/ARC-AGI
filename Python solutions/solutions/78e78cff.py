def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a background value (most common), pattern cells, and a unique marker cell
    2. Output creates a cross-shaped fill with the marker value through the marker position
    3. The cross has different thicknesses based on the pattern's bounding box dimensions
    4. Pattern cells are always preserved in the output

    Procedure:
    1. Identify background (most common), marker (appears once), and pattern cells
    2. Find pattern bounding box to determine row_extent and col_extent
    3. Determine fill thicknesses based on which extent is larger
    4. Apply vertical fill (certain columns for all rows)
    5. Apply horizontal fill (certain rows, fully or partially based on pattern cells)
    6. Always preserve pattern cells
    """
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Find background (most common value)
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    value_counts = Counter(all_values)
    background = value_counts.most_common(1)[0][0]

    # Find marker (value that appears exactly once)
    marker = None
    marker_pos = None
    for val, count in value_counts.items():
        if count == 1:
            marker = val
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == val:
                        marker_pos = (r, c)
                        break
                if marker_pos:
                    break
            break

    if marker is None:
        return [row[:] for row in grid]

    # Find pattern cells (non-background, non-marker)
    pattern_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and grid[r][c] != marker:
                pattern_cells.append((r, c))

    if not pattern_cells:
        return [row[:] for row in grid]

    # Compute pattern bounding box
    row_min = min(r for r, c in pattern_cells)
    row_max = max(r for r, c in pattern_cells)
    col_min = min(c for r, c in pattern_cells)
    col_max = max(c for r, c in pattern_cells)

    row_extent = row_max - row_min + 1
    col_extent = col_max - col_min + 1

    marker_row, marker_col = marker_pos

    # Determine thicknesses based on pattern dimensions
    if row_extent > col_extent:
        # Pattern is taller: thin vertical fill, thick horizontal fill
        vertical_thickness = 1
        horizontal_thickness = row_extent // 2 + 1
    else:
        # Pattern is wider or equal: thick vertical fill, thin horizontal fill
        vertical_thickness = col_extent // 2
        horizontal_thickness = row_extent // 2 + 1

    # Compute vertical fill range (columns to fill)
    col_start = marker_col - (vertical_thickness - 1) // 2
    col_end = marker_col + vertical_thickness // 2

    # Compute horizontal fill range (rows to fill)
    pattern_center_row = (row_min + row_max) / 2.0
    row_start = int(pattern_center_row - (horizontal_thickness - 1) / 2.0)
    row_end = row_start + horizontal_thickness - 1

    # Create output grid
    result = [row[:] for row in grid]

    # Apply vertical fill: fill certain columns for all rows
    for r in range(rows):
        for c in range(col_start, col_end + 1):
            if 0 <= c < cols:
                # Only fill if it's background or marker, preserve pattern cells
                if grid[r][c] == background or grid[r][c] == marker:
                    result[r][c] = marker

    # Apply horizontal fill: fill certain rows (fully or partially)
    for r in range(row_start, row_end + 1):
        if 0 <= r < rows:
            # Find pattern cells in this row
            pattern_cols_in_row = [c for c in range(cols)
                                   if grid[r][c] != background and grid[r][c] != marker]

            if not pattern_cols_in_row:
                # No pattern cells: fill entire row
                for c in range(cols):
                    if grid[r][c] == background or grid[r][c] == marker:
                        result[r][c] = marker
            else:
                # Has pattern cells: fill between them or to edges
                left_most = min(pattern_cols_in_row)
                right_most = max(pattern_cols_in_row)

                if left_most < right_most:
                    # Pattern cells on both sides: fill between them
                    for c in range(left_most + 1, right_most):
                        if grid[r][c] == background or grid[r][c] == marker:
                            result[r][c] = marker
                else:
                    # Pattern cell only on left: fill from it to right edge
                    for c in range(left_most + 1, cols):
                        if grid[r][c] == background or grid[r][c] == marker:
                            result[r][c] = marker

    return result
