def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid with non-zero values scattered throughout and markers (5s) in the bottom row.
    2. The output transforms marked columns by creating vertical segments filled with specific values.
    3. Columns marked with 5 in the bottom row are the ones to be processed.
    4. Each non-zero value in a marked column defines a segment that fills downward to create vertical bands.
    5. The grid is segmented vertically where each value occupies a contiguous range of rows in its column.

    Procedure:
    1. Identify columns marked with 5 in the bottom row.
    2. For each marked column, collect all non-zero values and their row positions.
    3. Sort the values by their row positions from top to bottom.
    4. Fill segments where the first value fills from row 0 to its position.
    5. Each subsequent value fills from after the previous value's position to its own position.
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find columns marked with 5 in bottom row
    for c in range(cols):
        if grid[rows - 1][c] == 5:
            # Collect non-zero values and their positions (excluding the marker 5)
            values_and_positions = []
            for r in range(rows - 1):  # Exclude bottom row marker
                if grid[r][c] != 0:
                    values_and_positions.append((r, grid[r][c]))

            # Add the bottom marker
            values_and_positions.append((rows - 1, 5))

            # Fill segments - each value extends from its position down to the next value's position
            if values_and_positions:
                # Sort by position (should already be sorted)
                values_and_positions.sort()

                # Process each value and determine its range
                for i, (pos, val) in enumerate(values_and_positions):
                    if i == 0:
                        # First value fills from top to its position
                        start = 0
                        end = pos
                    else:
                        # Subsequent values fill from after previous position to their position
                        start = values_and_positions[i - 1][0] + 1
                        end = pos

                    # Fill the range
                    for r in range(start, end + 1):
                        result[r][c] = val

    return result
