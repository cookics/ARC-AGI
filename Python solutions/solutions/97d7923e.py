def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with vertical segments of values
    2. A "segment" is: anchor value X at top, different uniform value Y in middle, same X at bottom
    3. Segments are grouped by (anchor, middle_value) pair
    4. For each group: if any column has non-zero above segment → change leftmost, else → change rightmost

    Procedure:
    1. Find all valid segments in each column
    2. Group segments by (anchor, middle_value)
    3. For each group, determine which column to transform based on above rule
    4. Replace middle values with anchor in selected columns
    """
    height = len(grid)
    width = len(grid[0])
    result = [row[:] for row in grid]

    # Find all valid segments: (col, start_row, end_row, anchor, middle)
    segments = []
    for col in range(width):
        if grid[height - 1][col] == 0:
            continue

        anchor = grid[height - 1][col]

        # Find topmost occurrence of anchor
        start_row = None
        for row in range(height - 2, -1, -1):
            if grid[row][col] == anchor:
                start_row = row

        if start_row is None:
            continue

        # Check middle has uniform different non-zero value
        middle_vals = [grid[row][col] for row in range(start_row + 1, height - 1)]
        if not middle_vals:
            continue
        if not all(v != 0 and v != anchor for v in middle_vals):
            continue
        if not all(v == middle_vals[0] for v in middle_vals):
            continue

        middle = middle_vals[0]
        segments.append((col, start_row, height - 1, anchor, middle))

    # Group segments by (anchor, middle)
    groups = {}
    for seg in segments:
        col, start_row, end_row, anchor, middle = seg
        key = (anchor, middle)
        if key not in groups:
            groups[key] = []
        groups[key].append(seg)

    # For each group, decide which column to transform
    for key, group in groups.items():
        # Check if any column has non-zero values above the segment
        has_nonzero_above = False
        for col, start_row, end_row, anchor, middle in group:
            for row in range(start_row):
                if grid[row][col] != 0:
                    has_nonzero_above = True
                    break
            if has_nonzero_above:
                break

        # Select column to transform
        if has_nonzero_above:
            # Change leftmost column
            target_col, target_start, target_end, target_anchor, _ = min(group, key=lambda x: x[0])
        else:
            # Change rightmost column
            target_col, target_start, target_end, target_anchor, _ = max(group, key=lambda x: x[0])

        # Replace middle values with anchor
        for row in range(target_start + 1, target_end):
            result[row][target_col] = target_anchor

    return result
