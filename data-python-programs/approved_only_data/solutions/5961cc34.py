def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has 1-patterns with 3s marking openings
    2. Column marker at position of value 4
    3. If 1 component: fill only marker column
    4. If multiple: activate components touching marker, extend via 3s, fill activated ones

    Procedure:
    1. Find components and marker column
    2. Activate components touching marker
    3. For activated components, extend via 3-openings
    4. Extensions activate components they touch
    5. Fill activated components and marker column
    """
    rows, cols = len(grid), len(grid[0])

    # Find marker column
    marker_col = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                marker_col = c
                break
        if marker_col is not None:
            break

    # Find components
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, comp):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 1:
            return
        visited[r][c] = True
        comp.add((r, c))
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            dfs(r + dr, c + dc, comp)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                comp = set()
                dfs(r, c, comp)
                components.append(comp)

    # Single component case
    if len(components) <= 1:
        result = [[8] * cols for _ in range(rows)]
        for r in range(rows):
            result[r][marker_col] = 2
        return result

    # Find activated components (touch marker)
    activated = set()
    for i, comp in enumerate(components):
        if any(c == marker_col for r, c in comp):
            activated.add(i)

    # Extensions storage
    extensions = set()

    # Apply extensions from activated components
    changed = True
    while changed:
        changed = False
        for comp_idx in list(activated):
            comp = components[comp_idx]

            # Find 3s adjacent to this component
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 3:
                        is_adjacent = any((r + dr, c + dc) in comp
                                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)])
                        if not is_adjacent:
                            continue

                        # Determine extension direction
                        has_1_left = (r, c-1) in comp if c > 0 else False
                        has_1_right = (r, c+1) in comp if c < cols-1 else False
                        has_1_above = (r-1, c) in comp if r > 0 else False
                        has_1_below = (r+1, c) in comp if r < rows-1 else False

                        # Horizontal extensions
                        if has_1_left and not has_1_right:
                            # RIGHT edge - extend right
                            # Find rightmost 1 in this row (across all components)
                            all_cols_in_row = [c2 for comp2 in components for r2, c2 in comp2 if r2 == r]
                            comp_leftmost = min((c2 for r2, c2 in comp if r2 == r), default=c)
                            # Check if there are 1s to the right of the 3
                            cols_to_right = [c2 for c2 in all_cols_in_row if c2 > c]
                            if not cols_to_right:
                                # No 1s to the right, extend to grid edge
                                for c2 in range(comp_leftmost, cols):
                                    extensions.add((r, c2))
                            else:
                                # Extend to rightmost 1 in row
                                rightmost = max(all_cols_in_row)
                                for c2 in range(comp_leftmost, rightmost + 1):
                                    extensions.add((r, c2))
                        elif has_1_right and not has_1_left:
                            # LEFT edge - extend left
                            # Find leftmost 1 in this row (across all components)
                            all_cols_in_row = [c2 for comp2 in components for r2, c2 in comp2 if r2 == r]
                            comp_rightmost = max((c2 for r2, c2 in comp if r2 == r), default=c)
                            # Check if there are 1s to the left of the 3
                            cols_to_left = [c2 for c2 in all_cols_in_row if c2 < c]
                            if not cols_to_left:
                                # No 1s to the left, extend to grid edge
                                for c2 in range(0, comp_rightmost + 1):
                                    extensions.add((r, c2))
                            else:
                                # Extend to leftmost 1 in row
                                leftmost = min(all_cols_in_row)
                                for c2 in range(leftmost, comp_rightmost + 1):
                                    extensions.add((r, c2))

                        # Vertical extensions
                        if has_1_below and not has_1_above:
                            # TOP edge - extend up
                            # Find all rows with 1s in this column (across all components)
                            all_rows_in_col = [r2 for comp2 in components for r2, c2 in comp2 if c2 == c]
                            comp_bottommost = max((r2 for r2, c2 in comp if c2 == c), default=r)
                            # Check if there are 1s above the 3
                            rows_above = [r2 for r2 in all_rows_in_col if r2 < r]
                            if not rows_above:
                                # No 1s above, extend to grid top
                                for r2 in range(0, comp_bottommost + 1):
                                    extensions.add((r2, c))
                            else:
                                # Extend to topmost 1 in column
                                topmost = min(all_rows_in_col)
                                for r2 in range(topmost, comp_bottommost + 1):
                                    extensions.add((r2, c))
                        elif has_1_above and not has_1_below:
                            # BOTTOM edge - extend down
                            # Find all rows with 1s in this column (across all components)
                            all_rows_in_col = [r2 for comp2 in components for r2, c2 in comp2 if c2 == c]
                            comp_topmost = min((r2 for r2, c2 in comp if c2 == c), default=r)
                            # Check if there are 1s below the 3
                            rows_below = [r2 for r2 in all_rows_in_col if r2 > r]
                            if not rows_below:
                                # No 1s below, extend to grid bottom
                                for r2 in range(comp_topmost, rows):
                                    extensions.add((r2, c))
                            else:
                                # Extend to bottommost 1 in column
                                bottommost = max(all_rows_in_col)
                                for r2 in range(comp_topmost, bottommost + 1):
                                    extensions.add((r2, c))

            # Check if extensions touch new components
            for i, comp in enumerate(components):
                if i not in activated:
                    if any(pos in extensions for pos in comp):
                        activated.add(i)
                        changed = True

    # Build output
    result = [[8] * cols for _ in range(rows)]

    # Fill activated components
    for comp_idx in activated:
        for r, c in components[comp_idx]:
            result[r][c] = 2

    # Fill extensions
    for r, c in extensions:
        result[r][c] = 2

    # Fill marker column from topmost activated component row to bottom
    min_comp_row = rows
    for comp_idx in activated:
        for r, c in components[comp_idx]:
            min_comp_row = min(min_comp_row, r)

    for r in range(min_comp_row, rows):
        result[r][marker_col] = 2

    return result
