def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a staircase pattern of 5s on the left (cols 0-8) growing from top to bottom
    2. Right section (cols 9-19) has a frame of 5s with interior values at odd columns (11, 13, 15, 17)
    3. Output clears the left staircase and transfers it to the right interior
    4. When 5s are placed in the right, non-5 values are compressed upward

    Procedure:
    1. Copy input to output
    2. Clear left section (cols 0-8) to all 0s
    3. For each odd column in left (1, 3, 5, 7):
       - Count how many 5s appear in that column (staircase height)
       - Find corresponding right column (col + 10)
       - Extract non-zero, non-5 values from right column
       - Determine processing region (from row 1 to last staircase row)
       - Keep only first (region_size - staircase_height) non-5 values
       - Place these at top of region, fill rest with 5s
    """

    rows = len(grid)
    cols = len(grid[0])
    output = [row[:] for row in grid]

    # Clear left section (cols 0-8)
    for r in range(rows):
        for c in range(9):
            output[r][c] = 0

    # Find the staircase region - last row with any 5 in left section
    last_staircase_row = 0
    for r in range(rows):
        for c in range(1, 9, 2):  # Check odd columns 1, 3, 5, 7
            if r < cols and c < len(grid[r]) and grid[r][c] == 5:
                last_staircase_row = max(last_staircase_row, r)

    # Processing region for right interior (typically rows 1 to last_staircase_row)
    first_interior_row = 1
    processing_rows = last_staircase_row - first_interior_row + 1

    if processing_rows > 0:
        # For each interior column in left section (both odd and even)
        for left_col in range(1, 8):
            # Identify which rows have 5s in this left column
            staircase_rows = []
            for r in range(first_interior_row, last_staircase_row + 1):
                if left_col < len(grid[r]) and grid[r][left_col] == 5:
                    staircase_rows.append(r)

            # Corresponding right column
            right_col = left_col + 10

            if right_col < cols and staircase_rows:
                # Extract non-zero, non-5 values from input
                non_five_values = []
                for r in range(first_interior_row, last_staircase_row + 1):
                    val = grid[r][right_col]
                    if val != 0 and val != 5:
                        non_five_values.append(val)

                # Number of non-5 values to keep
                num_to_keep = processing_rows - len(staircase_rows)
                kept_values = non_five_values[:num_to_keep]

                # Determine where to place kept values
                # If not enough kept values, fill from bottom of non-staircase region
                non_staircase_rows = [r for r in range(first_interior_row, last_staircase_row + 1) if r not in staircase_rows]

                if len(kept_values) >= len(non_staircase_rows):
                    # Enough values to fill all non-staircase rows, fill from top
                    kept_idx = 0
                    for r in range(first_interior_row, last_staircase_row + 1):
                        if r in staircase_rows:
                            output[r][right_col] = 5
                        else:
                            output[r][right_col] = kept_values[kept_idx]
                            kept_idx += 1
                else:
                    # Not enough values, keep original values at top, fill displaced values at bottom
                    # Place kept values starting from the bottom of non-staircase region
                    start_row = non_staircase_rows[-len(kept_values)] if kept_values else None
                    kept_idx = 0
                    for r in range(first_interior_row, last_staircase_row + 1):
                        if r in staircase_rows:
                            output[r][right_col] = 5
                        elif start_row is not None and r >= start_row:
                            output[r][right_col] = kept_values[kept_idx]
                            kept_idx += 1
                        # else: keep original value (already in output)

    return output
