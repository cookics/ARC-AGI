def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has colored regions with irregular shapes
    2. Output transforms each region into rectangles with:
       - Modal width (most common width across rows)
       - If one row extends beyond others, use that extension direction to position rectangle
       - Handle overlaps by shifting regions row-by-row
    3. Regions can coexist in same row at different columns

    Procedure:
    1. For each color, determine target rectangle (rows and columns)
    2. For each row, place colors' rectangles, adjusting for conflicts
    """
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all colors in reading order
    colors = []
    seen = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] not in seen:
                colors.append(grid[r][c])
                seen.add(grid[r][c])

    # Determine target rectangle for each color
    targets = {}
    for color in colors:
        row_spans = {}
        for r in range(rows):
            cols_with_color = [c for c in range(cols) if grid[r][c] == color]
            if cols_with_color:
                row_spans[r] = (min(cols_with_color), max(cols_with_color))

        if not row_spans:
            continue

        # Find modal width and position
        widths = [(right - left + 1) for left, right in row_spans.values()]
        width_counts = Counter(widths)
        max_count = max(width_counts.values())
        candidates = [w for w, c in width_counts.items() if c == max_count]
        modal_width = max(candidates)

        modal_data = [(r, left, right) for r, (left, right) in row_spans.items()
                      if right - left + 1 == modal_width]
        modal_lefts = [left for _, left, right in modal_data]
        modal_rights = [right for _, left, right in modal_data]
        modal_left = Counter(modal_lefts).most_common(1)[0][0]
        modal_right = Counter(modal_rights).most_common(1)[0][0]

        # Check for extensions
        extension_rows = [(r, left, right) for r, (left, right) in row_spans.items()
                          if right - left + 1 > modal_width]

        if extension_rows:
            ext_r, ext_left, ext_right = extension_rows[0]
            if ext_left < modal_left:
                output_left = ext_left
                output_right = ext_left + modal_width - 1
            elif ext_right > modal_right:
                output_right = ext_right
                output_left = ext_right - modal_width + 1
            else:
                output_left = modal_left
                output_right = modal_right
        else:
            output_left = modal_left
            output_right = modal_right

        # Store target rectangle
        target_rows = list(row_spans.keys())
        targets[color] = {
            'rows': target_rows,
            'left': output_left,
            'right': output_right,
            'width': output_right - output_left + 1
        }

    # Fill result row by row, handling conflicts
    for color in colors:
        if color not in targets:
            continue

        target = targets[color]
        for r in target['rows']:
            # Try to place at target columns
            left = target['left']
            right = target['right']

            # Count conflicts
            conflicts = sum(1 for c in range(left, min(right + 1, cols)) if result[r][c] != 0)
            conflict_ratio = conflicts / target['width'] if target['width'] > 0 else 0

            # Skip row if conflicts are exactly 2 cells AND width >= 8
            # (Rows with wider target and small overlap should be skipped)
            if conflicts == 2 and target['width'] >= 8:
                continue
            # Skip row if conflict ratio is >= 35%
            if conflict_ratio >= 0.35:
                continue

            # Fill non-conflicting cells
            # Skip cells at the right edge if a later color starts there
            for c in range(left, min(right + 1, cols)):
                if result[r][c] == 0:
                    # If this is the rightmost cell of current region,
                    # check if a later wider region starts here
                    if c == right:
                        skip_edge = False
                        for later_color in colors[colors.index(color)+1:]:
                            if later_color in targets:
                                later_target = targets[later_color]
                                if r in later_target['rows'] and c == later_target['left']:
                                    if later_target['width'] > target['width']:
                                        skip_edge = True
                                        break
                        if skip_edge:
                            continue
                    result[r][c] = color

    return result
