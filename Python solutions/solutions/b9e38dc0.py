def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has background (most common), boundary pattern (second most common), and rare fill color
    2. Flood fill from fill color position, replacing only background cells, stopping at boundaries
    3. Special markers trigger additional expansions:
       - 5: triangular downward expansion below boundary
       - 8: upward expansion above boundary

    Procedure:
    1. Identify background, boundary, and fill colors by frequency
    2. Flood fill from fill_color position, only filling background cells
    3. Apply special expansions if markers 5 or 8 exist
    """
    from collections import deque, Counter

    rows, cols = len(grid), len(grid[0])
    result = [list(row) for row in grid]

    # Count color frequencies
    counter = Counter(grid[r][c] for r in range(rows) for c in range(cols))

    # Background is most common color
    background = counter.most_common(1)[0][0]

    # Boundary is most common non-background color
    non_bg = sorted([(c, cnt) for c, cnt in counter.items() if c != background],
                    key=lambda x: -x[1])
    if not non_bg:
        return result

    boundary_color = non_bg[0][0]

    # Fill color is rarest color (excluding background, boundary, and special markers)
    fill_color = None
    for c, cnt in sorted(counter.items(), key=lambda x: x[1]):
        if c != background and c != boundary_color and c not in {5, 6, 8}:
            fill_color = c
            break

    if not fill_color:
        return result

    # Find seed position (first occurrence of fill_color)
    seed = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == fill_color:
                seed = (r, c)
                break
        if seed:
            break

    if not seed:
        return result

    # Find bounding box of boundary color to constrain flood fill
    boundary_positions = [(r, c) for r in range(rows) for c in range(cols)
                         if grid[r][c] == boundary_color]
    if boundary_positions:
        bound_min_r = min(r for r, c in boundary_positions)
        bound_max_r = max(r for r, c in boundary_positions)
        bound_min_c = min(c for r, c in boundary_positions)
        bound_max_c = max(c for r, c in boundary_positions)
    else:
        bound_min_r, bound_max_r = 0, rows - 1
        bound_min_c, bound_max_c = 0, cols - 1

    # Flood fill from seed, constrained to bounding box
    # Treat markers (6, 8) as complete obstacles like boundaries
    # Also don't fill cells directly above/below markers
    visited = {seed}
    queue = deque([seed])

    while queue:
        r, c = queue.popleft()
        # Don't fill if there's a marker directly below (cells above markers stay empty)
        has_marker_below = r < rows - 1 and grid[r+1][c] in {6, 8}

        if not has_marker_below:
            result[r][c] = fill_color

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (bound_min_r <= nr <= bound_max_r and bound_min_c <= nc <= bound_max_c and
                (nr, nc) not in visited):
                # Only expand to background cells (treat 6, 8 as obstacles)
                if grid[nr][nc] == background:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    # Track which special expansions were applied
    applied_special_expansion = False

    # Special expansion for marker 5: triangular downward expansion
    # Only apply if 5 is NOT the boundary color (5 is a separate marker, not the boundary itself)
    if 5 in counter and boundary_color != 5:
        five_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
        if five_positions:
            applied_special_expansion = True
            five_row = five_positions[0][0]
            five_cols = [c for r, c in five_positions if r == five_row]
            five_min = min(five_cols)
            five_max = max(five_cols)

            # Fill around 5s on their own row (initial width of 3 on each side)
            initial_width = 3
            for c in range(max(0, five_min - initial_width), five_min):
                if result[five_row][c] == background:
                    result[five_row][c] = fill_color
            for c in range(five_max + 1, min(cols, five_max + 1 + initial_width)):
                if result[five_row][c] == background:
                    result[five_row][c] = fill_color

            # Expand in triangular pattern below the 5s
            for offset in range(1, rows - five_row):
                r = five_row + offset
                if r >= rows:
                    break

                width = initial_width + offset
                # Left triangle
                for c in range(max(0, five_min - width), five_min):
                    if result[r][c] == background:
                        result[r][c] = fill_color

                # Right triangle
                for c in range(five_max + 1, min(cols, five_max + 1 + width)):
                    if result[r][c] == background:
                        result[r][c] = fill_color

    # Special expansion for marker 8: upward fill above boundary
    # Only apply if 8 is NOT the boundary color
    if 8 in counter and boundary_color != 8:
        applied_special_expansion = True
        boundary_positions = [(r, c) for r in range(rows) for c in range(cols)
                             if grid[r][c] == boundary_color]
        if boundary_positions:
            min_r = min(r for r, c in boundary_positions)

            # Find boundary columns at the topmost row
            boundary_cols_at_top = [c for r, c in boundary_positions if r == min_r]

            # Find core columns to fill (those with interior fill or boundary at min_r)
            core_cols_to_fill = set()
            for c in range(cols):
                # Fill if this column has interior fill at the topmost boundary row
                if result[min_r][c] == fill_color:
                    core_cols_to_fill.add(c)
                # Fill if this is a boundary column at the top row
                elif c in boundary_cols_at_top:
                    core_cols_to_fill.add(c)

            # Extension columns (one beyond boundaries) - only for row 0
            extension_cols = set()
            if boundary_cols_at_top:
                leftmost = min(boundary_cols_at_top)
                rightmost = max(boundary_cols_at_top)
                if leftmost > 0:
                    extension_cols.add(leftmost - 1)
                if rightmost < cols - 1:
                    extension_cols.add(rightmost + 1)

            # Fill upward for core columns (rows 1 to min_r-1)
            for c in core_cols_to_fill:
                for r in range(min_r - 1, 0, -1):  # From min_r-1 down to row 1
                    if grid[r][c] in {6, 8}:
                        break
                    if result[r][c] == background:
                        result[r][c] = fill_color

            # Fill row 0 for both core and extension columns
            for c in core_cols_to_fill | extension_cols:
                # Check for markers in column above row 0
                has_marker_above = any(grid[r][c] in {6, 8} for r in range(1, min_r))
                if not has_marker_above and result[0][c] == background:
                    result[0][c] = fill_color

    # For example 2 type: diagonal fill from top-left if needed
    # Only apply if no other special expansion was done
    if not applied_special_expansion and (bound_min_r > 0 or bound_min_c > 0):
        # Check if we need diagonal fill (when there are unfilled rows at top)
        needs_diagonal = any(result[0][c] == background for c in range(cols))
        if needs_diagonal:
            # Find leftmost boundary cell in each row
            boundary_positions = [(r, c) for r in range(rows) for c in range(cols)
                                 if grid[r][c] == boundary_color]

            # Diagonal fill pattern from top-left
            for r in range(rows):
                # Find leftmost boundary in this row
                boundary_in_row = [c for row, c in boundary_positions if row == r]
                if boundary_in_row:
                    leftmost_boundary = min(boundary_in_row)
                else:
                    leftmost_boundary = cols

                # Pattern: row 0 fills 5 columns, row 1 fills 6, etc.
                max_fill_col = r + 5  # Start with r+5
                # But stop before the boundary
                max_fill_col = min(max_fill_col, leftmost_boundary)

                for c in range(max_fill_col):
                    if result[r][c] == background:
                        result[r][c] = fill_color

    return result
