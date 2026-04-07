def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has separator lines forming a lattice structure
    2. Markers at separator intersections are encoded into 3x3 output
    3. Key insight: Process each separator row independently
       - Each separator row maps to an output row
       - Within that row, consecutive marker runs determine fill pattern

    Procedure:
    1. Find separator rows/columns
    2. For each separator row, extract marker runs
    3. Map runs to output cells based on position and length
    """

    from collections import Counter, defaultdict

    rows, cols = len(grid), len(grid[0])

    # Find separator value (most common non-zero)
    all_vals = [grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    sep_val = Counter(all_vals).most_common(1)[0][0]

    # Find separator rows and columns
    sep_rows = [r for r in range(rows) if sum(1 for c in range(cols) if grid[r][c] != 0) >= cols * 0.7]
    sep_cols = [c for c in range(cols) if sum(1 for r in range(rows) if grid[r][c] != 0) >= rows * 0.7]

    num_sep_rows = len(sep_rows)
    num_sep_cols = len(sep_cols)

    result = [[0] * 3 for _ in range(3)]

    # Process each separator row
    for sep_row_idx, sep_row in enumerate(sep_rows):
        # Extract markers in this separator row
        row_markers = []
        for sep_col_idx, sep_col in enumerate(sep_cols):
            val = grid[sep_row][sep_col]
            if val != 0 and val != sep_val:
                row_markers.append((sep_col_idx, val))

        if not row_markers:
            continue

        # Find consecutive runs of same value
        runs = []
        i = 0
        while i < len(row_markers):
            start_col_idx, val = row_markers[i]
            run_length = 1

            # Extend run while consecutive and same value
            j = i + 1
            while j < len(row_markers):
                next_col_idx, next_val = row_markers[j]
                if next_col_idx == row_markers[j-1][0] + 1 and next_val == val:
                    run_length += 1
                    j += 1
                else:
                    break

            runs.append((val, start_col_idx, run_length))
            i = j

        # Map this separator row to output row
        out_row = sep_row_idx * 3 // num_sep_rows

        # Process each run
        for val, start_col_idx, run_length in runs:
            # Number of output cells to fill: run_length - 1
            cells_to_fill = run_length - 1

            # Starting output column based on position
            out_col_start = start_col_idx * 3 // num_sep_cols

            # Fill cells
            for i in range(min(cells_to_fill, 3 - out_col_start)):
                out_col = out_col_start + i
                if out_col < 3 and result[out_row][out_col] == 0:
                    result[out_row][out_col] = val

    return result
