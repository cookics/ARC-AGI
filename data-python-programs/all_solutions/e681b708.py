def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has separator lines (vertical/horizontal) made of continuous non-zeros
    2. Separators have markers (2,3,4,6,8) at specific positions
    3. Non-separator 1s in each region get replaced by the nearest appropriate marker
    4. Preference: most common marker in the region's bounding corners

    Procedure:
    1. Identify separator lines
    2. For each non-separator 1, find all markers on its bounding separators
    3. Choose marker based on frequency, then prefer bottom/right positions
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all separator positions (rows and columns)
    separator_rows = set()
    separator_cols = set()

    # Detect separator columns (must have mostly non-zeros)
    for c in range(cols):
        non_zero_count = sum(1 for r in range(rows) if grid[r][c] != 0)
        if non_zero_count > rows // 2:  # More than 50% non-zero
            separator_cols.add(c)

    # Detect separator rows (must have mostly non-zeros)
    for r in range(rows):
        non_zero_count = sum(1 for c in range(cols) if grid[r][c] != 0)
        if non_zero_count > cols // 2:  # More than 50% non-zero
            separator_rows.add(r)

    # Helper to check if cell is on separator
    def is_on_separator(r, c):
        return r in separator_rows or c in separator_cols

    # Helper to find all markers on separators near a cell
    def find_region_markers(r, c):
        # Find bounding separators
        sep_above = max([sr for sr in separator_rows if sr < r], default=None)
        sep_below = min([sr for sr in separator_rows if sr > r], default=None)
        sep_left = max([sc for sc in separator_cols if sc < c], default=None)
        sep_right = min([sc for sc in separator_cols if sc > c], default=None)

        # Collect markers at the 4 corners of the region
        marker_data = []

        # Top-left
        if sep_above is not None and sep_left is not None:
            if grid[sep_above][sep_left] not in [0, 1]:
                marker_data.append((grid[sep_above][sep_left], False, False))

        # Top-right
        if sep_above is not None and sep_right is not None:
            if grid[sep_above][sep_right] not in [0, 1]:
                marker_data.append((grid[sep_above][sep_right], False, True))

        # Bottom-left
        if sep_below is not None and sep_left is not None:
            if grid[sep_below][sep_left] not in [0, 1]:
                marker_data.append((grid[sep_below][sep_left], True, False))

        # Bottom-right
        if sep_below is not None and sep_right is not None:
            if grid[sep_below][sep_right] not in [0, 1]:
                marker_data.append((grid[sep_below][sep_right], True, True))

        return marker_data

    # Transform each isolated 1
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not is_on_separator(r, c):
                marker_data = find_region_markers(r, c)

                if marker_data:
                    from collections import Counter
                    markers = [m for m, _, _ in marker_data]
                    counts = Counter(markers)
                    max_count = max(counts.values())

                    # Get candidates with max frequency
                    candidates = [m for m, count in counts.items() if count == max_count]

                    if len(candidates) == 1:
                        result[r][c] = candidates[0]
                    else:
                        # Tiebreaker: prefer bottom, then right
                        best_marker = None
                        best_priority = (-1, -1)
                        for marker, is_below, is_right in marker_data:
                            if marker in candidates:
                                priority = (is_below, is_right)
                                if priority > best_priority:
                                    best_priority = priority
                                    best_marker = marker
                        result[r][c] = best_marker

    return result
