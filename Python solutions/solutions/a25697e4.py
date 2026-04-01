def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Background is most frequent color
    2. Anchor shape (most cells) stays fixed
    3. Other shapes move to cluster around anchor
    4. Gaps in anchor bbox are filled; remaining shapes go adjacent
    5. Far-apart shapes are brought together with column shift

    Procedure:
    1. Find anchor (most cells) - stays in place
    2. Find gaps in anchor's bounding box
    3. Fill gaps with cells from one other shape
    4. Place remaining shapes adjacent to anchor
    """

    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter
    color_count = Counter()
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] += 1

    background = color_count.most_common(1)[0][0]

    # Extract shapes
    shapes = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in shapes:
                    shapes[color] = []
                shapes[color].append((r, c))

    if not shapes:
        return [row[:] for row in grid]

    # Find anchor
    anchor_color = max(shapes.keys(), key=lambda c: len(shapes[c]))
    anchor_cells = set(shapes[anchor_color])

    # Get anchor bounding box
    anchor_rows = [r for r, c in anchor_cells]
    anchor_cols = [c for r, c in anchor_cells]
    anchor_min_r, anchor_max_r = min(anchor_rows), max(anchor_rows)
    anchor_min_c, anchor_max_c = min(anchor_cols), max(anchor_cols)

    # Find gaps in anchor bbox
    gaps = []
    for r in range(anchor_min_r, anchor_max_r + 1):
        for c in range(anchor_min_c, anchor_max_c + 1):
            if (r, c) not in anchor_cells:
                gaps.append((r, c))

    # Initialize result
    result = [[background for _ in range(cols)] for _ in range(rows)]

    # Place anchor
    for r, c in anchor_cells:
        result[r][c] = anchor_color

    # Process other shapes
    other_colors = [c for c in shapes.keys() if c != anchor_color]

    if not other_colors:
        return result

    # Check if shapes are far apart
    all_other_cells = []
    for color in other_colors:
        all_other_cells.extend(shapes[color])

    other_min_c = min(c for r, c in all_other_cells)
    other_max_c = max(c for r, c in all_other_cells)
    other_center_c = (other_min_c + other_max_c) / 2
    anchor_center_c = (anchor_min_c + anchor_max_c) / 2

    if abs(other_center_c - anchor_center_c) > 10:
        # Far apart - bring together by filling gaps and placing adjacent

        # Fill gaps - use all gaps for one shape
        if gaps and other_colors:
            gap_filler = min(other_colors, key=lambda c: abs(len(shapes[c]) - len(gaps)))
            for r, c in gaps:
                result[r][c] = gap_filler

        # Place remaining shapes adjacent to anchor
        for color in other_colors:
            if gaps and color == gap_filler:
                continue

            # Get this shape's cells and find its bounding box
            color_cells = shapes[color]
            color_rows = sorted(set(r for r, c in color_cells))
            color_cols = sorted(set(c for r, c in color_cells))

            # Map shape cells to a compact grid starting at (0,0)
            color_min_r, color_max_r = min(color_rows), max(color_rows)
            color_min_c, color_max_c = min(color_cols), max(color_cols)

            # If gaps exist, start at the first gap row; otherwise at anchor's top
            if gaps:
                start_r = min(r for r, c in gaps)
            else:
                start_r = anchor_min_r
            start_c = anchor_max_c + 1

            for r, c in color_cells:
                # Calculate relative position within the shape
                rel_r = r - color_min_r
                rel_c = c - color_min_c

                # Place at new position
                new_r = start_r + rel_r
                new_c = start_c + rel_c

                if 0 <= new_r < rows and 0 <= new_c < cols:
                    if result[new_r][new_c] == background:
                        result[new_r][new_c] = color

    else:
        # Close together - proportional movement toward anchor
        anchor_center_r = sum(anchor_rows) / len(anchor_rows)

        for color in other_colors:
            color_cells = shapes[color]
            color_rows = [r for r, c in color_cells]
            color_cols = [c for r, c in color_cells]
            color_center_r = sum(color_rows) / len(color_rows)
            color_center_c = sum(color_cols) / len(color_cols)

            shift_r = int((anchor_center_r - color_center_r) / 2)
            shift_c = int((anchor_center_c - color_center_c) * 0.8)

            for r, c in color_cells:
                new_r = r + shift_r
                new_c = c + shift_c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    if result[new_r][new_c] == background:
                        result[new_r][new_c] = color

    return result
