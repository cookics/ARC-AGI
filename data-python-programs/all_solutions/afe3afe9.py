def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with a line of 1s on one edge (top/bottom row or left/right column)
    2. Grid contains 3x3 hollow square patterns (value on border, 0 in center)
    3. Two different encoding schemes:
       - Horizontal 1s (top/bottom): count-based encoding with alignment
       - Vertical 1s (left/right): position-based encoding with cumulative overlay
    4. Each row group (3 consecutive rows with patterns) becomes one output row

    Procedure:
    1. Find the line of 1s to determine which encoding scheme to use
    2. Identify all row groups containing 3x3 patterns
    3. Apply appropriate encoding scheme based on 1s position
    """

    rows, cols = len(grid), len(grid[0])

    # Find line of 1s to determine encoding scheme
    use_position_encoding = False
    align_right = False

    if all(grid[0][j] == 1 for j in range(cols)):  # Top row
        use_position_encoding = False
        align_right = True
    elif all(grid[rows-1][j] == 1 for j in range(cols)):  # Bottom row
        use_position_encoding = False
        align_right = False
    elif all(grid[i][cols-1] == 1 for i in range(rows)):  # Right column
        use_position_encoding = True
        align_right = True
    elif all(grid[i][0] == 1 for i in range(rows)):  # Left column
        use_position_encoding = False  # Use count-based like top/bottom
        align_right = False

    # Find row groups with 3x3 patterns
    row_groups = []
    r = 1
    while r < rows - 2:
        has_pattern = False
        for c in range(1, cols - 2):
            if (grid[r][c] != 0 and grid[r][c] != 1 and
                grid[r][c+1] != 0 and grid[r][c+2] != 0 and
                grid[r+1][c] != 0 and grid[r+1][c+1] == 0 and grid[r+1][c+2] != 0 and
                grid[r+2][c] != 0 and grid[r+2][c+1] != 0 and grid[r+2][c+2] != 0 and
                grid[r][c] == grid[r][c+2] == grid[r+1][c] == grid[r+1][c+2] ==
                grid[r+2][c] == grid[r+2][c+1] == grid[r+2][c+2] == grid[r][c+1]):
                has_pattern = True
                break

        if has_pattern:
            row_groups.append((r, r+1, r+2))
            r += 3
        else:
            r += 1

    # Extract pattern positions and colors for each row group
    def find_patterns(r_start, r_mid, r_end):
        patterns = []
        c = 1
        while c < cols - 2:
            if (grid[r_start][c] != 0 and grid[r_start][c] != 1 and
                grid[r_start][c+1] != 0 and grid[r_start][c+2] != 0 and
                grid[r_mid][c] != 0 and grid[r_mid][c+1] == 0 and grid[r_mid][c+2] != 0 and
                grid[r_end][c] != 0 and grid[r_end][c+1] != 0 and grid[r_end][c+2] != 0):

                color = grid[r_start][c]
                if (color == grid[r_start][c+1] == grid[r_start][c+2] ==
                    grid[r_mid][c] == grid[r_mid][c+2] ==
                    grid[r_end][c] == grid[r_end][c+1] == grid[r_end][c+2]):

                    patterns.append((c, color))
                    c += 4
                    continue
            c += 1
        return patterns

    # Process based on encoding scheme
    if not use_position_encoding:
        # Count-based encoding (top/bottom/left 1s)
        pattern_info = []
        for r_start, r_mid, r_end in row_groups:
            patterns = find_patterns(r_start, r_mid, r_end)

            left_patterns = [(c, col) for c, col in patterns if c < 13]
            right_patterns = [(c, col) for c, col in patterns if c > 16]

            left_color = left_patterns[0][1] if left_patterns else 0
            right_color = right_patterns[0][1] if right_patterns else 0
            left_count = len(left_patterns)
            right_count = len(right_patterns)

            pattern_info.append((left_count, left_color, right_count, right_color))

        max_width = max(left_cnt + right_cnt for left_cnt, _, right_cnt, _ in pattern_info) if pattern_info else 0

        result = []
        for left_cnt, left_col, right_cnt, right_col in pattern_info:
            row = [left_col] * left_cnt + [right_col] * right_cnt

            while len(row) < max_width:
                if align_right:
                    row.insert(0, 0)
                else:
                    row.append(0)

            result.append(row)

        return result

    else:
        # Position-based encoding (right column 1s) with cumulative overlay
        # Map column positions to output positions
        # Left: cols 1,5,9 → pos 0,1,2; separator pos 3; Right: cols 17,21,25 → pos 4,5,6
        col_to_pos = {1: 0, 5: 1, 9: 2, 17: 4, 21: 5, 25: 6}
        output_width = 7

        # Build result with cumulative overlay and visibility rules
        result_grid = [[0] * output_width for _ in range(len(row_groups))]

        for row_idx, (r_start, r_mid, r_end) in enumerate(row_groups):
            # First, inherit from previous row
            if row_idx > 0:
                result_grid[row_idx] = result_grid[row_idx - 1][:]

            # Then overlay current group's patterns with visibility rules
            patterns = find_patterns(r_start, r_mid, r_end)
            left_cols = [c for c, _ in patterns if c in [1, 5, 9]]
            right_cols = [c for c, _ in patterns if c in [17, 21, 25]]

            # Apply visibility rules for first row
            if row_idx == 0:
                # First row: show only boundary right patterns
                for c, color in patterns:
                    if c in [17, 25]:  # Only boundary right positions
                        result_grid[row_idx][col_to_pos[c]] = color
            else:
                # Other rows: mark positions where patterns exist
                for c, color in patterns:
                    if c in col_to_pos:
                        result_grid[row_idx][col_to_pos[c]] = color

            # Fill separator on last row if all other positions have same non-zero color
            if row_idx == len(row_groups) - 1:
                non_sep_values = [result_grid[row_idx][i] for i in range(output_width) if i != 3]
                if len(set(non_sep_values)) == 1 and non_sep_values[0] != 0:
                    result_grid[row_idx][3] = non_sep_values[0]

        return result_grid
