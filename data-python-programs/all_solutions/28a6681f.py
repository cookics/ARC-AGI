def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has two triangular regions of main colors (not 0, not 1) growing from different areas
    2. Input has 1s scattered, some need to be removed, some need to fill gaps
    3. Main colors are identified from the bottom row (excluding 0 and 1)
    4. For each row:
       - If no main colors present, remove all 1s
       - If main colors present but 1s are far (distance > 4), remove 1s
       - Otherwise, fill gaps between non-zero values with 1s

    Procedure:
    1. Identify main colors from bottom row
    2. For each row, check main color presence
    3. Remove 1s that are too far from main colors
    4. Fill 0s between non-zero values (if they match or one is a 1)
    """

    # Identify main colors from the bottom row
    bottom_row = grid[-1]
    main_colors = set(c for c in bottom_row if c != 0 and c != 1)

    result = [row[:] for row in grid]  # Copy grid

    for r in range(len(grid)):
        row = grid[r]

        # Find main colors in this row
        main_color_positions = [c for c, val in enumerate(row) if val in main_colors]

        if not main_color_positions:
            # No main colors, remove all 1s
            for c in range(len(row)):
                if row[c] == 1:
                    result[r][c] = 0
        else:
            # At least one main color present
            right_main = max(main_color_positions)

            # Find 1s in this row
            one_positions = [c for c, val in enumerate(row) if val == 1]

            # Check if we should remove 1s
            should_remove_ones = False
            if one_positions:
                left_one = min(one_positions)

                # Check distance from rightmost main color to leftmost 1
                if left_one > right_main:
                    distance = left_one - right_main
                    if distance > 4:
                        should_remove_ones = True

            if should_remove_ones:
                # Remove 1s
                for c in one_positions:
                    result[r][c] = 0
            else:
                # Fill gaps
                # For each 0, check if it's between two non-0 values that are either the same or one is a 1
                for c in range(len(row)):
                    if row[c] == 0:
                        # Find the nearest non-0 value to the left
                        left_val = None
                        for i in range(c - 1, -1, -1):
                            if row[i] != 0:
                                left_val = row[i]
                                break

                        # Find the nearest non-0 value to the right
                        right_val = None
                        for i in range(c + 1, len(row)):
                            if row[i] != 0:
                                right_val = row[i]
                                break

                        # Check if we should fill this 0
                        if left_val is not None and right_val is not None:
                            if left_val == right_val or left_val == 1 or right_val == 1:
                                result[r][c] = 1

    return result
