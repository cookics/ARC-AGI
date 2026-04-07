def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a vertical segment of 9s (length L)
    2. Output moves 9s to a different column with L+1 cells
    3. The target column has a vertical segment of 8s with length 2*(L+1)
    4. This segment is split: half becomes 9s, half stays 8s
    5. Original 9s are replaced with 8s

    Procedure:
    1. Find vertical segment of 9s (column, start row, end row, length L)
    2. Find first column (left to right) with vertical segment of 8s of length >= 2*(L+1)
    3. Compare midpoints: if 9s_midpoint <= 8s_midpoint, put 9s in first half; else second half
    4. Replace original 9s with 8s
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find vertical segment of 9s
    nine_col = None
    nine_start = None
    nine_end = None

    for c in range(cols):
        for r in range(rows):
            if grid[r][c] == 9:
                if nine_col is None:
                    nine_col = c
                    nine_start = r
                    nine_end = r
                elif c == nine_col:
                    nine_end = r

    if nine_col is None:
        return result

    nine_length = nine_end - nine_start + 1
    nine_midpoint = (nine_start + nine_end) / 2.0
    required_length = 2 * (nine_length + 1)

    # Find first suitable column with vertical segment of 8s
    target_col = None
    target_start = None
    target_end = None

    for c in range(cols):
        if c == nine_col:
            continue

        # Find vertical segments of 8s in this column
        start = None
        for r in range(rows):
            if grid[r][c] == 8:
                if start is None:
                    start = r
            else:
                if start is not None:
                    length = r - start
                    if length >= required_length:
                        target_col = c
                        target_start = start
                        target_end = start + required_length - 1
                        break
                start = None

        # Check if segment extends to end of grid
        if start is not None and target_col is None:
            length = rows - start
            if length >= required_length:
                target_col = c
                target_start = start
                target_end = start + required_length - 1

        if target_col is not None:
            break

    if target_col is None:
        return result

    # Replace original 9s with 8s
    for r in range(nine_start, nine_end + 1):
        result[r][nine_col] = 8

    # Determine where to place 9s in target segment
    target_midpoint = (target_start + target_end) / 2.0
    half_length = (nine_length + 1)

    if nine_midpoint <= target_midpoint:
        # Place 9s in first half
        for r in range(target_start, target_start + half_length):
            result[r][target_col] = 9
    else:
        # Place 9s in second half
        for r in range(target_end - half_length + 1, target_end + 1):
            result[r][target_col] = 9

    return result
