def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with 0s and 1s forming various patterns
    2. Output is the same grid with some 0s replaced by 2s
    3. Horizontal bars are rows with 5+ consecutive 1s that serve as boundaries
    4. There's a vertical "spine" column where 1s are consistently present
    5. 2s are added to the LEFT of the spine in columns [1, spine-1]
    6. Specific fill pattern depends on row characteristics and number of horizontal bars

    Procedure:
    1. Find horizontal bars (rows with 5+ consecutive 1s) and identify reference row
    2. Determine fill region boundaries (rows around reference row)
    3. Find the spine column (leftmost column with 1 in reference row)
    4. For each row in fill region, determine which columns to fill based on row pattern
    5. Fill appropriate columns with 2s (always avoiding column 0)
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find max consecutive 1s in each row
    row_consecutive = []
    for r in range(rows):
        max_consecutive = 0
        consecutive = 0
        for c in range(cols):
            if grid[r][c] == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        row_consecutive.append(max_consecutive)

    # Find horizontal bars (rows with 5+ consecutive 1s)
    horizontal_bars = [r for r in range(rows) if row_consecutive[r] >= 5]

    if not horizontal_bars:
        return result

    # Find reference row (row with maximum consecutive 1s, prefer middle)
    max_consecutive = max(row_consecutive)
    candidates = [r for r in range(rows) if row_consecutive[r] == max_consecutive]

    # Pick reference row based on position relative to bars
    middle_candidates = []
    for r in candidates:
        bars_above = [b for b in horizontal_bars if b < r]
        bars_below = [b for b in horizontal_bars if b > r]
        if bars_above and bars_below:
            middle_candidates.append(r)

    if middle_candidates:
        reference_row = middle_candidates[len(middle_candidates) // 2]
    elif len(horizontal_bars) == 1:
        reference_row = horizontal_bars[0]
    else:
        reference_row = candidates[0]

    # Determine fill region based on number of horizontal bars
    if len(horizontal_bars) == 1:
        # Single bar: fill from reference row downward
        fill_start = reference_row
        fill_end = reference_row + 2
    else:
        # Multiple bars: fill around reference row
        fill_start = reference_row - 2
        fill_end = reference_row + 2

    # Adjust boundaries to avoid empty rows at edges
    while fill_start < reference_row and row_consecutive[fill_start] == 0:
        fill_start += 1
    while fill_end > reference_row and row_consecutive[fill_end] == 0:
        fill_end -= 1

    # Find spine column (leftmost column with 1 in reference row)
    spine_col = None
    for c in range(cols):
        if grid[reference_row][c] == 1:
            spine_col = c
            break

    if spine_col is None:
        return result

    # Determine fillable columns: [1, spine-1]
    fillable_cols = list(range(1, spine_col))

    # Apply fill patterns for each row in fill region
    for r in range(max(0, fill_start), min(rows, fill_end + 1)):
        if row_consecutive[r] == 0:
            continue

        # Determine which columns to fill based on row characteristics
        cols_to_fill = []

        if r == reference_row:
            # Reference row: fill based on number of horizontal bars
            if len(horizontal_bars) == 1:
                # Single bar: fill half of fillable range
                cols_to_fill = fillable_cols[:min(2, len(fillable_cols))]
            else:
                # Multiple bars: fill all fillable columns
                cols_to_fill = fillable_cols
        else:
            # Check for gap pattern (1,0,1)
            has_gap = False
            for c in range(cols - 2):
                if grid[r][c] == 1 and grid[r][c + 1] == 0 and grid[r][c + 2] == 1:
                    has_gap = True
                    break

            if has_gap:
                # Gap pattern: different fill based on bars count
                if len(horizontal_bars) == 1:
                    # Single bar: fill first 3 columns of fillable range
                    cols_to_fill = fillable_cols[:min(3, len(fillable_cols))]
                else:
                    # Multiple bars: fill cols at positions 0 and 2 of fillable range
                    if len(fillable_cols) >= 3:
                        cols_to_fill = [fillable_cols[0], fillable_cols[2]]
                    else:
                        cols_to_fill = fillable_cols
            elif row_consecutive[r] in [1, 3]:
                # Rows with 1 or 3 consecutive 1s: fill only first column
                if fillable_cols:
                    cols_to_fill = [fillable_cols[0]]
            else:
                # Default: fill first 2 columns of fillable range
                cols_to_fill = fillable_cols[:min(2, len(fillable_cols))]

        # Apply the fill
        for c in cols_to_fill:
            if result[r][c] == 0:
                result[r][c] = 2

    return result
