def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains solid 3×3 blocks that need to convert to + patterns
    2. Input may have existing cross patterns (+ or X shape)
    3. Patterns are paired/mirrored between regions with color transformations
    4. Background elements should be preserved during transformation

    Procedure:
    1. Identify background vs object colors
    2. Find solid 3×3 blocks and convert to + patterns
    3. Find existing cross patterns
    4. Apply mirroring/swapping logic based on spatial arrangement
    """

    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Identify background colors (very frequent or forms connected regions)
    color_counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    # Background colors are typically 0 or very frequent structural elements
    background_colors = {0}
    # Add very frequent colors that likely form structure
    total_cells = rows * cols
    for color, count in color_counts.items():
        if count > total_cells * 0.3:  # More than 30% is likely background
            background_colors.add(color)

    def is_background(val):
        return val in background_colors

    def make_plus(r, c, color):
        if not (1 <= r < rows-1 and 1 <= c < cols-1):
            return
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if is_background(result[r+dr][c+dc]):
                    result[r+dr][c+dc] = 0
        result[r-1][c] = result[r+1][c] = result[r][c-1] = result[r][c+1] = color

    # Find solid 3×3 blocks
    solids = []
    for r in range(rows-2):
        for c in range(cols-2):
            val = grid[r][c]
            if not is_background(val):
                if all(grid[r+dr][c+dc] == val for dr in range(3) for dc in range(3)):
                    center = (r+1, c+1)
                    if not any(abs(center[0]-sr) <= 1 and abs(center[1]-sc) <= 1
                              for sr, sc, _ in solids):
                        solids.append((center[0], center[1], val))

    # Convert solids to +
    for r, c, val in solids:
        make_plus(r, c, val)

    # Find existing crosses (+ or X patterns)
    crosses = []
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if any(r == sr and c == sc for sr, sc, _ in solids):
                continue

            # Check + pattern
            arms = [grid[r-1][c], grid[r+1][c], grid[r][c-1], grid[r][c+1]]
            if not is_background(arms[0]) and all(a == arms[0] for a in arms):
                if grid[r][c] == 0:
                    crosses.append((r, c, arms[0]))
                    continue

            # Check X pattern
            val = grid[r][c]
            if not is_background(val):
                diags = [grid[r-1][c-1], grid[r-1][c+1], grid[r+1][c-1], grid[r+1][c+1]]
                if all(d == val for d in diags):
                    make_plus(r, c, val)
                    crosses.append((r, c, val))

    if not solids and not crosses:
        return result

    all_objects = solids + crosses
    all_cols = sorted(set(c for _, c, _ in all_objects))

    # Group by row (use average spacing between objects to determine grouping)
    all_rows = [r for r, _, _ in all_objects]
    row_spacing = []
    for i in range(len(all_rows) - 1):
        row_spacing.append(abs(all_rows[i+1] - all_rows[i]))

    # Determine grouping threshold dynamically
    if row_spacing:
        avg_spacing = sum(row_spacing) / len(row_spacing)
        group_threshold = max(2, avg_spacing * 0.5)
    else:
        group_threshold = 2

    row_groups = {}
    for r, c, val in all_objects:
        found = False
        for key in sorted(row_groups.keys()):
            if abs(key - r) <= group_threshold:
                row_groups[key].append((r, c, val, 'solid' if (r, c, val) in solids else 'cross'))
                found = True
                break
        if not found:
            row_groups[r] = [(r, c, val, 'solid' if (r, c, val) in solids else 'cross')]

    sorted_row_keys = sorted(row_groups.keys())

    # Case 1: Single row band - create symmetric mirror
    if len(sorted_row_keys) == 1:
        top_row_key = sorted_row_keys[0]
        objects = sorted(row_groups[top_row_key], key=lambda x: x[1])

        # Find empty region for mirroring
        min_clear_row = None
        search_start = rows - 2
        search_end = top_row_key + 2

        for r in range(search_start, search_end, -1):
            if r < 1 or r >= rows - 1:
                continue
            is_clear = True
            for _, c_obj, _, _ in objects:
                # Check if we can place patterns near same columns
                for test_col in range(max(1, c_obj - 1), min(cols - 1, c_obj + 1)):
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            test_r = r + dr
                            test_c = test_col + dc
                            if 0 <= test_r < rows and 0 <= test_c < cols:
                                if not is_background(result[test_r][test_c]):
                                    is_clear = False
                                    break
                        if not is_clear:
                            break
                    if not is_clear:
                        break
                if not is_clear:
                    break
            if is_clear:
                min_clear_row = r
                break

        if min_clear_row:
            colors = [val for _, _, val, _ in objects]
            cols_used = [c for _, c, _, _ in objects]

            # Reverse color order for mirror
            reversed_colors = colors[::-1]

            # Place mirrored patterns with small offset
            for i, orig_col in enumerate(cols_used):
                if i < len(reversed_colors):
                    # Try original column first, then nearby
                    for offset in [0, -1, 1]:
                        target_col = orig_col + offset
                        if 1 <= target_col < cols - 1:
                            make_plus(min_clear_row, target_col, reversed_colors[i])
                            break

    # Case 2: Multiple rows - handle color swapping and duplication
    elif len(sorted_row_keys) >= 2:
        # Collect all unique columns per region
        col_usage = {}
        for r, c, val, typ in all_objects:
            if c not in col_usage:
                col_usage[c] = []
            col_usage[c].append((r, val, typ))

        # Determine left/right columns
        all_cols_sorted = sorted(col_usage.keys())
        if len(all_cols_sorted) >= 2:
            mid_col = (all_cols_sorted[0] + all_cols_sorted[-1]) / 2.0
            left_cols = [c for c in all_cols_sorted if c < mid_col]
            right_cols = [c for c in all_cols_sorted if c > mid_col]
        else:
            return result

        # Build map: (row_key, col) -> (val, type)
        obj_map = {}
        for row_key in sorted_row_keys:
            for r, c, val, typ in row_groups[row_key]:
                obj_map[(row_key, c)] = (val, typ)

        # Process each row-column combination
        for row_key in sorted_row_keys:
            row_objs = row_groups[row_key]

            # Check what columns are used in this row
            cols_in_row = [c for _, c, _, _ in row_objs]

            # If this row only has objects in left columns
            if all(c in left_cols for c in cols_in_row):
                # Find what columns other rows use in right region
                for other_row in sorted_row_keys:
                    if other_row == row_key:
                        continue
                    other_cols = [c for _, c, _, _ in row_groups[other_row] if c in right_cols]
                    if other_cols:
                        # Place swapped colors at those columns
                        for lc in left_cols:
                            if (row_key, lc) in obj_map:
                                _, _, val_here, typ_here = [(r,c,v,t) for r,c,v,t in row_objs if c == lc][0]
                                # Find what color to swap with
                                for other_lc in left_cols:
                                    if (other_row, other_lc) in obj_map and other_lc == lc:
                                        other_val, other_typ = obj_map[(other_row, other_lc)]
                                        # Place swapped color in right columns
                                        for rc in other_cols:
                                            if (row_key, rc) not in obj_map:
                                                make_plus(row_key, rc, other_val)
                                                obj_map[(row_key, rc)] = (other_val, 'created')
                                                break
                                        break
                        break

            # If this row only has objects in right columns
            elif all(c in right_cols for c in cols_in_row):
                # Find what columns other rows use in left region
                for other_row in sorted_row_keys:
                    if other_row == row_key:
                        continue
                    other_cols = [c for _, c, _, _ in row_groups[other_row] if c in left_cols]
                    if other_cols:
                        # Place swapped/duplicated colors at those columns
                        for rc in right_cols:
                            if (row_key, rc) in obj_map:
                                _, _, val_here, typ_here = [(r,c,v,t) for r,c,v,t in row_objs if c == rc][0]
                                # Duplicate same color to left
                                for lc in other_cols:
                                    if (row_key, lc) not in obj_map:
                                        make_plus(row_key, lc, val_here)
                                        obj_map[(row_key, lc)] = (val_here, 'created')
                                        break
                        break

    return result
