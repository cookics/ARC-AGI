def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 19x19 grid with a few non-zero cells forming a small pattern
    2. Output creates a symmetric frame by reflecting these cells 4-way around center (9,9)
    3. After reflection, values that appear multiple times on same row/column need edge filling
    4. Edges are filled at alternating positions (every 2 steps) between repeated values

    Procedure:
    1. Reflect all non-zero cells to create 4-fold symmetry
    2. For each row, find values appearing multiple times and fill between them
    3. For each column, find values appearing multiple times and fill between them
    4. Fill at alternating positions (step=2) between min and max positions of each value
    """
    from collections import Counter

    # Create result grid
    result = [[0] * len(grid[0]) for _ in range(len(grid))]

    # Reflect all non-zero cells 4-way
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                val = grid[r][c]
                result[r][c] = val
                result[r][18-c] = val
                result[18-r][c] = val
                result[18-r][18-c] = val

    # Find rows with >= 4 cells, but only fill the outermost ones
    rows_with_4plus = []
    for r in range(len(result)):
        cells = [(c, result[r][c]) for c in range(len(result[0])) if result[r][c] != 0]
        if len(cells) >= 4:
            rows_with_4plus.append((r, cells))

    # Only fill the min and max rows that have >= 4 cells
    if rows_with_4plus:
        min_row = min(r for r, cells in rows_with_4plus)
        max_row = max(r for r, cells in rows_with_4plus)
        rows_to_fill = [(r, cells) for r, cells in rows_with_4plus if r == min_row or r == max_row]
    else:
        rows_to_fill = []

    # Find columns with >= 4 cells, but only fill the outermost ones
    cols_with_4plus = []
    for c in range(len(result[0])):
        cells = [(r, result[r][c]) for r in range(len(result)) if result[r][c] != 0]
        if len(cells) >= 4:
            cols_with_4plus.append((c, cells))

    # Only fill the min and max columns that have >= 4 cells
    if cols_with_4plus:
        min_col = min(c for c, cells in cols_with_4plus)
        max_col = max(c for c, cells in cols_with_4plus)
        cols_to_fill = [(c, cells) for c, cells in cols_with_4plus if c == min_col or c == max_col]
    else:
        cols_to_fill = []

    # Fill selected rows
    for r, cells in rows_to_fill:
        all_cols = [c for c, v in cells]
        min_col = min(all_cols)
        max_col = max(all_cols)

        value_counts = Counter(v for c, v in cells)
        for val, count in value_counts.items():
            if count >= 2:
                positions = [c for c, v in cells if v == val]
                if len(positions) >= 2:
                    val_min = min(positions)
                    val_max = max(positions)
                    # Only fill if this value is NOT at the extreme positions
                    if val_min != min_col or val_max != max_col:
                        for c in range(val_min+2, val_max, 2):
                            if result[r][c] == 0:
                                result[r][c] = val

    # Fill selected columns
    for c, cells in cols_to_fill:
        all_rows = [r for r, v in cells]
        min_row = min(all_rows)
        max_row = max(all_rows)

        value_counts = Counter(v for r, v in cells)
        for val, count in value_counts.items():
            if count >= 2:
                positions = [r for r, v in cells if v == val]
                if len(positions) >= 2:
                    val_min = min(positions)
                    val_max = max(positions)
                    # Only fill if this value is NOT at the extreme positions
                    if val_min != min_row or val_max != max_row:
                        for r in range(val_min+2, val_max, 2):
                            if result[r][c] == 0:
                                result[r][c] = val

    return result
