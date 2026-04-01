def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with values: 0 (background), 1 (markers), and other values (scattered objects like 5 or 8)
    2. Output fills certain cells with 1 based on marker positions
    3. For each marker (value 1):
       - The entire row containing the marker gets filled with 1 (except existing non-zero values)
       - For each scattered value (non-zero, non-1) in the marker row, create a cross pattern (±1 row, ±1 column)
       - Special rule: If a row in the cross has a scattered value to the LEFT of the center column, skip that row
       - If a row has a scattered value at or to the RIGHT of center, fill it and extend around that scattered value
       - Extensions can propagate one level (rows with scattered values can trigger fills in adjacent rows)

    Procedure:
    1. Find all marker positions (cells with value 1)
    2. For each marker:
       a. Fill the entire marker row (preserve non-zero values)
       b. Find scattered values in the marker row
       c. For each scattered value, process ±1 rows with special rules
       d. Track and process scattered values found during filling (one level of extension)
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find all markers (value 1)
    markers = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    for row_m, col_m in markers:
        # Step 1: Identify vertically blocked positions and scattered values
        blocked = set()
        scattered_cols = []
        for c in range(cols):
            if grid[row_m][c] not in [0, 1]:
                scattered_cols.append(c)
            elif result[row_m][c] == 0:
                has_above = row_m - 1 >= 0 and grid[row_m - 1][c] not in [0, 1]
                has_below = row_m + 1 < rows and grid[row_m + 1][c] not in [0, 1]
                if has_above and has_below:
                    blocked.add(c)

        # Find small gaps between scattered values that contain blocked positions
        skip_positions = set()
        for i in range(len(scattered_cols) - 1):
            left_scatter = scattered_cols[i]
            right_scatter = scattered_cols[i + 1]
            gap_size = right_scatter - left_scatter - 1
            # Only skip small gaps (<=4) that contain blocked positions
            if gap_size <= 4:
                gap_has_blocked = any(
                    c in blocked for c in range(left_scatter + 1, right_scatter)
                )
                if gap_has_blocked:
                    # Skip the entire gap
                    skip_positions.update(range(left_scatter + 1, right_scatter))

        # Fill the marker row, skipping gaps with blocked positions
        for c in range(cols):
            if result[row_m][c] == 0 and c not in skip_positions:
                result[row_m][c] = 1

        # Step 2: Find scattered values in the marker row (non-zero, non-1)
        scattered_to_process = [
            (row_m, col) for col in range(cols) if grid[row_m][col] not in [0, 1]
        ]
        processed = set()

        # Track what's been queued to avoid duplicates
        queued = set(scattered_to_process)

        # Step 3: Process scattered values and their crosses
        while scattered_to_process:
            row_s, col_s = scattered_to_process.pop(0)
            if (row_s, col_s) in processed:
                continue
            processed.add((row_s, col_s))

            # Process rows ±1 from the scattered value
            for dr in [-1, 1]:
                r = row_s + dr
                if 0 <= r < rows and r != row_m:  # Don't reprocess marker row
                    # Check for scattered values in columns ±1 from col_s
                    cols_range = [
                        col_s + dc for dc in [-1, 0, 1] if 0 <= col_s + dc < cols
                    ]
                    scattered_in_row = [
                        (r, c) for c in cols_range if grid[r][c] not in [0, 1]
                    ]

                    # Special rule: Skip row if scattered value is to the LEFT of col_s
                    # BUT only if that scattered value hasn't been queued for processing
                    left_scattered = [
                        (r, c)
                        for c in cols_range
                        if c < col_s and grid[r][c] not in [0, 1]
                    ]
                    if left_scattered:
                        # Skip only if none of these left scattered values are already queued/processed
                        if not any(
                            pos in queued or pos in processed for pos in left_scattered
                        ):
                            continue

                    # Fill columns ±1 from col_s
                    for dc in [-1, 0, 1]:
                        c = col_s + dc
                        if 0 <= c < cols and result[r][c] == 0:
                            result[r][c] = 1

                    # If there are scattered values at or to the RIGHT, extend around them
                    for r_scat, c_scat in scattered_in_row:
                        if c_scat >= col_s:
                            # Extend columns ±1 from the scattered value
                            for dc in [-1, 0, 1]:
                                c = c_scat + dc
                                if 0 <= c < cols and result[r][c] == 0:
                                    result[r][c] = 1
                            # Add for further processing (one level of recursion)
                            if (r_scat, c_scat) not in queued and (
                                r_scat,
                                c_scat,
                            ) not in processed:
                                scattered_to_process.append((r_scat, c_scat))
                                queued.add((r_scat, c_scat))

    return result
