def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has spatial regions with dominant colors
    2. Accent colors (smaller count) within these regions are relocated
    3. Vertical projection upward when height >= width
    4. Horizontal projection left/right when width > height
    5. Number of cells placed equals span (row_span for vertical, col_span for horizontal)

    Procedure:
    1. Identify all colors and their positions
    2. For each color pair, check if smaller is accent within larger's bbox
    3. Remove accent cells and place them outside the dominant region
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Get all color positions
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((r, c))

    if len(color_positions) <= 1:
        return result

    # Track which positions have been processed
    processed_positions = set()

    # For each color pair, check if one is accent within the other
    colors = list(color_positions.keys())

    for accent_color in colors:
        for dominant_color in colors:
            if accent_color == dominant_color:
                continue

            accent_pos = color_positions[accent_color]
            dominant_pos = color_positions[dominant_color]

            # Check if accent is smaller (likely an accent)
            if len(accent_pos) >= len(dominant_pos):
                continue

            # Get bounding boxes
            dom_rows = [r for r, c in dominant_pos]
            dom_cols = [c for r, c in dominant_pos]

            dom_bbox = (min(dom_rows), max(dom_rows), min(dom_cols), max(dom_cols))

            # Find accent positions within dominant bbox that haven't been processed
            accent_in_bbox = []
            for r, c in accent_pos:
                if (r, c) not in processed_positions:
                    if dom_bbox[0] <= r <= dom_bbox[1] and dom_bbox[2] <= c <= dom_bbox[3]:
                        accent_in_bbox.append((r, c))

            if not accent_in_bbox:
                continue

            # Check if significant portion of remaining accent is in this bbox
            acc_rows = [r for r, c in accent_in_bbox]
            acc_cols = [c for r, c in accent_in_bbox]
            acc_bbox = (min(acc_rows), max(acc_rows), min(acc_cols), max(acc_cols))

            # Check overlap ratio
            overlap_rows = set(range(acc_bbox[0], acc_bbox[1] + 1)) & set(range(dom_bbox[0], dom_bbox[1] + 1))
            overlap_cols = set(range(acc_bbox[2], acc_bbox[3] + 1)) & set(range(dom_bbox[2], dom_bbox[3] + 1))

            if len(overlap_rows) == 0 or len(overlap_cols) == 0:
                continue

            overlap_ratio = (len(overlap_rows) / (acc_bbox[1] - acc_bbox[0] + 1)) * \
                           (len(overlap_cols) / (acc_bbox[3] - acc_bbox[2] + 1))

            if overlap_ratio < 0.4:
                continue

            # Clear these accent positions
            for r, c in accent_in_bbox:
                result[r][c] = 0
                processed_positions.add((r, c))

            # Determine projection direction
            height = dom_bbox[1] - dom_bbox[0] + 1
            width = dom_bbox[3] - dom_bbox[2] + 1

            # Calculate spans
            min_accent_row = min(acc_rows)
            max_accent_row = max(acc_rows)
            min_accent_col = min(acc_cols)
            max_accent_col = max(acc_cols)
            row_span = max_accent_row - min_accent_row + 1
            col_span = max_accent_col - min_accent_col + 1

            if height >= width:
                # Vertical projection - place just above dominant bbox
                num_rows = row_span

                # Find median column
                accent_cols_list = sorted([c for r, c in accent_in_bbox])
                median_col = accent_cols_list[len(accent_cols_list) // 2]

                # Place just above dominant bbox
                for i in range(num_rows):
                    target_row = dom_bbox[0] - num_rows + i
                    if 0 <= target_row < rows:
                        result[target_row][median_col] = accent_color
            else:
                # Horizontal projection
                accent_avg_col = sum(c for r, c in accent_in_bbox) / len(accent_in_bbox)
                dom_avg_col = sum(dom_cols) / len(dom_cols)

                if accent_avg_col < dom_avg_col:
                    # Accent on left, place on right: find row with rightmost dominant cell
                    best_row = None
                    rightmost_col = -1
                    for r in set(dom_rows):
                        if min_accent_row <= r <= max_accent_row:
                            row_dom_cols = [c for rr, c in dominant_pos if rr == r]
                            if row_dom_cols:
                                max_col_in_row = max(row_dom_cols)
                                if max_col_in_row > rightmost_col:
                                    rightmost_col = max_col_in_row
                                    best_row = r
                    if best_row is not None:
                        start_col = rightmost_col + 1
                        for i in range(col_span):
                            if start_col + i < cols:
                                result[best_row][start_col + i] = accent_color
                else:
                    # Accent on right, place on left: find row with leftmost dominant cell
                    best_row = None
                    leftmost_col = cols
                    for r in set(dom_rows):
                        if r >= max_accent_row:
                            row_dom_cols = [c for rr, c in dominant_pos if rr == r]
                            if row_dom_cols:
                                min_col_in_row = min(row_dom_cols)
                                if min_col_in_row < leftmost_col:
                                    leftmost_col = min_col_in_row
                                    best_row = r
                    if best_row is not None:
                        start_col = leftmost_col - col_span
                        for i in range(col_span):
                            if 0 <= start_col + i < cols:
                                result[best_row][start_col + i] = accent_color

    return result
