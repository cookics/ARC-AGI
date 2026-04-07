def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has exactly one cell with value 2 (the "anchor")
    2. Draw rectangular frames connecting anchor to other colored cells
    3. Key patterns:
       - Same-row cells: horizontal line + vertical extension (stops at non-zero cells in original grid)
       - Same-column cells: vertical at offset column + horizontal at closest row above anchor
       - Single-cell edge rows (above anchor only): horizontal + vertical with extensions
       - Key rows with 3+ spread cells: trigger full horizontal lines one row above
    4. Structures stop one cell before target and don't overwrite non-zero values

    Procedure:
    1. Find anchor and organize all colored cells
    2. Process anchor row structures with smart vertical extension
    3. Process same-column structures
    4. Process edge row structures (only above anchor, single cell)
    5. Process key rows with multiple spread cells
    6. Return modified grid
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find anchor
    anchor_r, anchor_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                anchor_r, anchor_c = r, c
                break
        if anchor_r is not None:
            break

    if anchor_r is None:
        return result

    # Organize cells by row and column
    cells_by_row = {}
    cells_by_col = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) != (anchor_r, anchor_c):
                if r not in cells_by_row:
                    cells_by_row[r] = []
                cells_by_row[r].append(c)

                if c not in cells_by_col:
                    cells_by_col[c] = []
                cells_by_col[c].append(r)

    # 1. Process anchor row
    anchor_row_rightmost = anchor_c
    if anchor_r in cells_by_row:
        rightmost_c = max(cells_by_row[anchor_r])
        anchor_row_rightmost = rightmost_c
        if rightmost_c > anchor_c:
            stop_c = rightmost_c - 1
            # Draw horizontal line
            for c in range(anchor_c, stop_c + 1):
                if result[anchor_r][c] == 0:
                    result[anchor_r][c] = 2

            # Draw vertical extension (stops at non-zero cells in original grid)
            # Extend upwards
            for r in range(anchor_r - 1, -1, -1):
                if grid[r][stop_c] != 0:
                    break
                if result[r][stop_c] == 0:
                    result[r][stop_c] = 2

            # Extend downwards
            for r in range(anchor_r, rows):
                if r > anchor_r and grid[r][stop_c] != 0:
                    break
                if result[r][stop_c] == 0:
                    result[r][stop_c] = 2

    # 2. Process same-column cells
    if anchor_c in cells_by_col:
        offset_c = anchor_c + 1 if anchor_c + 1 < cols else anchor_c - 1

        # Draw vertical line at offset (full height, skipping non-zero)
        for r in range(rows):
            if result[r][offset_c] == 0:
                result[r][offset_c] = 2

        # Draw horizontal at closest same-column row above anchor
        same_col_rows_above = [r for r in cells_by_col[anchor_c] if r < anchor_r]
        if same_col_rows_above:
            closest_row = max(same_col_rows_above)
            extent_c = (
                anchor_row_rightmost + 1
                if anchor_row_rightmost > anchor_c
                else cols - 1
            )
            for c in range(offset_c, min(extent_c + 1, cols)):
                if result[closest_row][c] == 0:
                    result[closest_row][c] = 2

    # 3. Process edge rows (only above anchor, single cell)
    elif cols - 1 in cells_by_col:
        for r in cells_by_col[cols - 1]:
            # Only process if row is above anchor and has exactly one cell at edge
            if r < anchor_r and len(cells_by_row.get(r, [])) == 1:
                stop_c = cols - 2
                # Draw horizontal line
                for c in range(anchor_c, stop_c + 1):
                    if result[r][c] == 0:
                        result[r][c] = 2

                # Draw vertical extension (stops at non-zero cells)
                # Extend upwards from r
                for row in range(r - 1, -1, -1):
                    if grid[row][stop_c] != 0:
                        break
                    if result[row][stop_c] == 0:
                        result[row][stop_c] = 2

                # Extend downwards from r
                for row in range(r, rows):
                    if row > r and grid[row][stop_c] != 0:
                        break
                    if result[row][stop_c] == 0:
                        result[row][stop_c] = 2
                break

    # 4. Find rows with value 6 - draw full-width horizontal lines
    for r in cells_by_row:
        for c in cells_by_row[r]:
            if grid[r][c] == 6:
                # Draw full-width horizontal line for this row
                for col in range(cols):
                    if result[r][col] == 0:
                        result[r][col] = 2
                break

    # 5. Find key rows with 3+ spread cells and create full horizontal lines one row above
    for r in sorted(cells_by_row.keys()):
        if r > anchor_r + 2 and len(cells_by_row[r]) >= 3:
            # Check if cells are spread across the width
            min_c = min(cells_by_row[r])
            max_c = max(cells_by_row[r])
            if max_c - min_c >= cols // 2:
                # Draw full horizontal line one row above
                target_row = r - 1
                for c in range(cols):
                    if result[target_row][c] == 0:
                        result[target_row][c] = 2
                break

    return result
