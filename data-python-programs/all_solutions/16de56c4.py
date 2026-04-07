def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with sparse non-zero values
    2. Output extends patterns either row-wise or column-wise
    3. If rows > cols: process each row independently
       - Find leftmost group of same values (defines spacing)
       - Find rightmost different value (defines fill value)
       - Extend pattern using spacing and fill value
    4. If cols >= rows: process columns based on last row
       - Last row indicates which columns to process
       - For each column, find value pattern and extend vertically

    Procedure:
    1. Determine if row-wise or column-wise processing
    2. For row-wise: process each row with non-zero values
    3. For column-wise: use last row as template, extend columns vertically
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if rows > cols:
        # Row-wise processing
        result = [row[:] for row in grid]  # Deep copy

        for r in range(rows):
            # Find all non-zero positions and values
            non_zero = [(c, grid[r][c]) for c in range(cols) if grid[r][c] != 0]

            if len(non_zero) < 2:
                continue

            # Group consecutive same values from left
            first_val = non_zero[0][1]
            pattern_positions = [pos for pos, val in non_zero if val == first_val]

            # Check if there's a different value (fill value)
            fill_val = first_val
            end_pos = cols - 1

            for pos, val in non_zero:
                if val != first_val:
                    fill_val = val
                    end_pos = pos
                    break

            # Determine spacing
            if len(pattern_positions) >= 2:
                spacing = pattern_positions[1] - pattern_positions[0]
            else:
                spacing = 1

            # Clear the row first
            result[r] = [0] * cols

            # Fill with pattern
            if spacing > 0:
                pos = pattern_positions[0]
                while pos <= end_pos:
                    result[r][pos] = fill_val
                    pos += spacing

    else:
        # Column-wise processing
        result = [[0] * cols for _ in range(rows)]

        # Get last row to determine which columns to process
        last_row = grid[-1]
        target_cols = [(c, last_row[c]) for c in range(cols) if last_row[c] != 0]

        for target_col, target_val in target_cols:
            # Find all non-zero positions in this column
            all_non_zero = [(r, grid[r][target_col]) for r in range(rows) if grid[r][target_col] != 0]

            if len(all_non_zero) < 2:
                # Just copy if only one non-zero
                for r, val in all_non_zero:
                    result[r][target_col] = val
                continue

            # Get positions only
            all_positions = [r for r, _ in all_non_zero]

            # Check if all positions have consistent spacing
            spacings = [all_positions[i+1] - all_positions[i] for i in range(len(all_positions)-1)]
            consistent = len(set(spacings)) == 1

            if consistent:
                # All non-zero positions have same spacing
                spacing = spacings[0]
                first_val = all_non_zero[0][1]

                # Extend pattern with first value
                start = all_positions[0] % spacing
                pos = start
                while pos < rows:
                    result[pos][target_col] = first_val
                    pos += spacing
            else:
                # Inconsistent spacing - use only target value positions
                target_positions = [r for r in range(rows) if grid[r][target_col] == target_val]

                if len(target_positions) >= 2:
                    spacing = target_positions[1] - target_positions[0]
                    start = target_positions[0] % spacing
                    pos = start
                    while pos < rows:
                        result[pos][target_col] = target_val
                        pos += spacing

                # Preserve other non-zero values
                for r, val in all_non_zero:
                    if val != target_val:
                        result[r][target_col] = val

        # Copy other non-zero values that weren't part of target columns
        target_col_set = {c for c, _ in target_cols}
        for r in range(rows):
            for c in range(cols):
                if c not in target_col_set and grid[r][c] != 0:
                    result[r][c] = grid[r][c]

    return result
