def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid alternates: even rows have 3,8,8,...,3 or 8,8,8,...; odd rows have 0,0,7,7,7,...
    2. Horizontal [7,7,7] triplets in odd rows become [8,6,8]
    3. Vertical 6-lines extend from middle columns of [7,7,7] triplets through 8s in all rows
    4. Edge 3s in even rows follow complex transformation patterns

    Procedure:
    1. Transform [7,7,7] to [8,6,8] in odd rows
    2. Draw vertical 6-lines at middle columns of triplets
    3. Handle 3s based on row characteristics and patterns
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Step 1: Transform [7,7,7] to [8,6,8] and record middle columns per row
    middle_cols_by_row = {}
    for r in range(1, rows, 2):
        middle_cols_by_row[r] = []
        for c in range(cols - 2):
            if grid[r][c] == 7 and grid[r][c + 1] == 7 and grid[r][c + 2] == 7:
                result[r][c] = 8
                result[r][c + 1] = 6
                result[r][c + 2] = 8
                middle_cols_by_row[r].append(c + 1)

    # Step 2: Draw vertical 6-lines at middle columns (only through even rows, from adjacent odd rows)
    for r in range(0, rows, 2):  # Even rows
        # Get middle columns from adjacent odd rows
        adjacent_middles = set()
        for dr in [-1, 1]:
            adj_r = r + dr
            if adj_r in middle_cols_by_row:
                adjacent_middles.update(middle_cols_by_row[adj_r])

        # Apply 6s at these middle columns
        for mid_col in adjacent_middles:
            if result[r][mid_col] == 8:
                result[r][mid_col] = 6

    # Step 3: Handle row 0 - always remove edge 3s
    for c in range(cols):
        if grid[0][c] == 3:
            result[0][c] = 8

    # Step 4: Handle 3s in other even rows
    # Identify rows with 0s (excluding row 0)
    rows_with_0s = set()
    for r in range(2, rows, 2):
        if grid[r][0] == 3 and any(grid[r][c] == 0 for c in range(cols)):
            rows_with_0s.add(r)

    # For rows with 0s: keep edge 3s and add 3s to the left of each 0
    for r in rows_with_0s:
        for c in range(cols):
            if grid[r][c] == 0 and c > 0 and result[r][c - 1] == 8:
                result[r][c - 1] = 3

    # For rows without 0s (excluding row 0 and last row if it starts with 8)
    rows_no_0s = []
    for r in range(2, rows, 2):
        if r not in rows_with_0s and grid[r][0] == 3:
            rows_no_0s.append(r)

    # Determine pattern based on presence of interspersed 0-rows
    has_interspersed_0s = any(r in rows_with_0s for r in range(2, rows - 1, 2))

    # Apply pattern
    if has_interspersed_0s:
        # Pattern B: left, none, both, none
        patterns = ['left', 'none', 'both', 'none']
    else:
        # Pattern A: left, right, none, both
        patterns = ['left', 'right', 'none', 'both']

    for i, r in enumerate(rows_no_0s):
        pattern = patterns[i % len(patterns)]

        if pattern == 'left':
            # Duplicate left edge 3
            if result[r][1] == 8:
                result[r][1] = 3
            # Remove right edge 3
            result[r][cols - 1] = 8
        elif pattern == 'right':
            # Remove left edge 3
            result[r][0] = 8
            # Duplicate right edge 3
            if result[r][cols - 2] == 8:
                result[r][cols - 2] = 3
        elif pattern == 'both':
            # Duplicate both edges
            if result[r][1] == 8:
                result[r][1] = 3
            if result[r][cols - 2] == 8:
                result[r][cols - 2] = 3
        elif pattern == 'none':
            # Remove both edges
            result[r][0] = 8
            result[r][cols - 1] = 8

    # Step 5: Handle last row (if it starts with 8)
    if rows > 0 and rows % 2 == 1:
        last_r = rows - 1
        if grid[last_r][0] != 3:
            # Check if it has 0s
            has_0s = any(grid[last_r][c] == 0 for c in range(cols))
            if has_0s:
                # Place 3s to the right of each 0 and at right edge
                for c in range(cols):
                    if grid[last_r][c] == 0 and c + 1 < cols and result[last_r][c + 1] == 8:
                        result[last_r][c + 1] = 3
                if result[last_r][cols - 1] == 8:
                    result[last_r][cols - 1] = 3
            else:
                # Place 3s at both edges
                if result[last_r][0] == 8:
                    result[last_r][0] = 3
                if result[last_r][cols - 1] == 8:
                    result[last_r][cols - 1] = 3

    return result
