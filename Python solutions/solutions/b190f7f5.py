def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains values 0-9 where 8 acts as a separator/background
    2. Non-0, non-8 values are expanded into n×n patterns in output
    3. Each input value creates either a cross or diagonal pattern
    4. Output size is (num_rows * num_active_cols)² (square)
    5. Pattern type depends on position of 8s relative to active value columns

    Procedure:
    1. Find active columns (columns containing non-0, non-8 values)
    2. Determine pattern type: if min(8_column) < min(active_column) → diagonal, else → cross
    3. Create output grid of size n*k × n*k where n=num_rows, k=num_active_cols
    4. For each non-0, non-8 value at (r,c), place n×n pattern at block position (r, col_idx)
    """

    if not grid or not grid[0]:
        return [[]]

    n_rows = len(grid)
    n_cols = len(grid[0])

    # Find active columns (columns with at least one non-0, non-8 value)
    active_cols = []
    for c in range(n_cols):
        has_value = any(grid[r][c] not in [0, 8] for r in range(n_rows))
        if has_value:
            active_cols.append(c)

    if not active_cols:
        return [[0]]

    # Map original column to active column index
    col_to_idx = {c: i for i, c in enumerate(active_cols)}

    # Determine pattern type by finding position of 8s
    min_8_col = float('inf')
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == 8:
                min_8_col = min(min_8_col, c)

    min_active_col = active_cols[0]
    use_diagonal = min_8_col < min_active_col

    # Create output grid
    k = len(active_cols)
    out_size = n_rows * k
    output = [[0] * out_size for _ in range(out_size)]

    # Generate pattern templates
    def get_cross_pattern(n):
        """Cross pattern: row 0 has center, row 1 has all, rest have center"""
        pattern = []
        center = n // 2
        for r in range(n):
            row = []
            for c in range(n):
                if r == 1:  # Row 1: all columns
                    row.append(True)
                elif c == center:  # Other rows: center column only
                    row.append(True)
                else:
                    row.append(False)
            pattern.append(row)
        return pattern

    def get_diagonal_pattern(n):
        """Diagonal pattern: forms an L-shape going down-left"""
        pattern = []
        for r in range(n):
            row = []
            for c in range(n):
                if r == 0:
                    # Row 0: rightmost column
                    row.append(c == n - 1)
                elif r == 1:
                    # Row 1: leftmost column
                    row.append(c == 0)
                else:
                    # Row 2+: columns 0 to (r-1)
                    row.append(c <= r - 1)
            pattern.append(row)
        return pattern

    # Select pattern
    pattern_template = get_diagonal_pattern(n_rows) if use_diagonal else get_cross_pattern(n_rows)

    # Place patterns for each non-0, non-8 value
    for r in range(n_rows):
        for c in range(n_cols):
            val = grid[r][c]
            if val != 0 and val != 8:
                col_idx = col_to_idx[c]
                base_row = r * n_rows
                base_col = col_idx * n_rows

                for dr in range(n_rows):
                    for dc in range(n_rows):
                        if pattern_template[dr][dc]:
                            output[base_row + dr][base_col + dc] = val

    return output
