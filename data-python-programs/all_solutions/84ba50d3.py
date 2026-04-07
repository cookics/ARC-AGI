def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a horizontal separator row filled with 2s dividing the grid
    2. Patterns above the separator get transformed column-by-column
    3. Each column's pattern is placed either above or below separator based on column position
    4. Left columns (< midpoint) go to one side, right columns (>= midpoint) go to the other
    5. The direction (which side) depends on grid geometry
    6. Separator is updated: 1 if column spans both sides, 8 if one side only

    Procedure:
    1. Find separator row
    2. Extract pattern from each column above separator
    3. Determine target placement based on column position and available space
    4. Place patterns in output
    5. Update separator row
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find separator
    sep_row = -1
    for r in range(rows):
        if all(grid[r][c] == 2 for c in range(cols)):
            sep_row = r
            break

    if sep_row == -1:
        return grid

    # Create output grid
    result = [[8] * cols for _ in range(rows)]
    result[sep_row] = [2] * cols

    # Calculate grid geometry
    rows_above_sep = sep_row
    rows_below_sep = rows - sep_row - 1
    mid_col = cols / 2.0

    # Count number of rows with patterns above separator
    pattern_rows = set()
    for r in range(sep_row):
        if any(grid[r][c] != 8 for c in range(cols)):
            pattern_rows.add(r)
    num_pattern_rows = len(pattern_rows)

    # Determine mapping direction based on pattern density
    # If many pattern rows (>2), use reverse: left above, right below
    # If few pattern rows (<=2), use normal: left below, right above
    use_reverse_mapping = (num_pattern_rows > 2)

    # Process each column
    for c in range(cols):
        # Extract non-8 cells from this column above separator
        pattern = []
        for r in range(sep_row):
            if grid[r][c] != 8:
                pattern.append((r, grid[r][c]))

        if not pattern:
            continue

        # Determine target side based on column position
        is_left_column = (c < mid_col)

        if use_reverse_mapping:
            # Reverse mapping: left goes above/spans, right goes below
            if is_left_column:
                # Place pattern spanning or above separator
                # If pattern is tall enough, span it
                if len(pattern) >= 3:
                    # Span across separator: place (len-1)/2 above and (len-1)/2 below
                    # The middle cell is absorbed by the separator
                    cells_per_side = (len(pattern) - 1) // 2

                    # Place cells above separator
                    for i in range(cells_per_side):
                        orig_r, val = pattern[i]
                        target_r = sep_row - cells_per_side + i
                        if 0 <= target_r < sep_row:
                            result[target_r][c] = val

                    # Place cells below separator
                    for i in range(cells_per_side):
                        orig_r, val = pattern[len(pattern) - cells_per_side + i]
                        target_r = sep_row + 1 + i
                        if target_r < rows:
                            result[target_r][c] = val
                else:
                    # Place above separator
                    start_r = sep_row - 1
                    for i in range(len(pattern) - 1, -1, -1):
                        orig_r, val = pattern[i]
                        target_r = start_r - (len(pattern) - 1 - i)
                        if 0 <= target_r < sep_row:
                            result[target_r][c] = val
            else:
                # Place pattern below separator
                start_r = rows - len(pattern)
                for i, (orig_r, val) in enumerate(pattern):
                    target_r = start_r + i
                    if sep_row < target_r < rows:
                        result[target_r][c] = val
        else:
            # Normal mapping: left goes below, right goes above
            if is_left_column:
                # Place pattern below separator, stacking from bottom
                start_r = rows - 1
                for i in range(len(pattern) - 1, -1, -1):
                    orig_r, val = pattern[i]
                    target_r = start_r - (len(pattern) - 1 - i)
                    if sep_row < target_r < rows:
                        result[target_r][c] = val
            else:
                # Place pattern above separator
                start_r = sep_row - 1
                for i in range(len(pattern) - 1, -1, -1):
                    orig_r, val = pattern[i]
                    target_r = start_r - (len(pattern) - 1 - i)
                    if 0 <= target_r < sep_row:
                        result[target_r][c] = val

    # Update separator row
    for c in range(cols):
        # Check if column had input above separator
        had_input_above = any(grid[r][c] != 8 for r in range(sep_row))

        if not had_input_above:
            continue  # stays as 2

        # Check where this column appears in output
        has_output_above = any(result[r][c] != 8 for r in range(sep_row))
        has_output_below = any(result[r][c] != 8 for r in range(sep_row + 1, rows))

        if has_output_above and has_output_below:
            # Column spans across separator
            result[sep_row][c] = 1
        elif has_output_below:
            # Column moved from input above to output below
            result[sep_row][c] = 8
        # else: column stayed above, separator stays as 2

    return result
