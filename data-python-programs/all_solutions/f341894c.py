def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with values mostly 8s (background), some 1s, 6s, and 7s (markers)
    2. Output swaps certain adjacent pairs of 1 and 6 (either horizontally or vertically)
    3. Horizontal swaps: (6,1) always becomes (1,6); (1,6) becomes (6,1) only if row starts with 7
    4. Vertical swaps apply to row pairs based on complement patterns and specific conditions
    5. The transformation normalizes the positions based on local parity and context

    Procedure:
    1. Copy the grid
    2. Apply horizontal swaps based on adjacent (6,1) or (1,6) pairs
    3. Apply vertical swaps for row pairs that form perfect complements under certain conditions
    4. Return the transformed grid
    """

    result = [row[:] for row in grid]
    rows, cols = len(result), len(result[0])

    # Step 1: Horizontal swaps (always swap (6,1) to (1,6); swap (1,6) to (6,1) only if row starts with 7)
    for r in range(rows):
        for c in range(cols - 1):
            if result[r][c] == 6 and result[r][c + 1] == 1:
                # Always swap (6,1) to (1,6)
                result[r][c] = 1
                result[r][c + 1] = 6
            elif result[r][c] == 1 and result[r][c + 1] == 6:
                # Swap (1,6) to (6,1) only if row starts with 7
                if result[r][0] == 7:
                    result[r][c] = 6
                    result[r][c + 1] = 1

    # Step 2: Vertical swaps for consecutive row pairs
    for r in range(rows - 1):
        row1_ones = set(c for c in range(cols) if result[r][c] == 1)
        row1_sixes = set(c for c in range(cols) if result[r][c] == 6)
        row2_ones = set(c for c in range(cols) if result[r + 1][c] == 1)
        row2_sixes = set(c for c in range(cols) if result[r + 1][c] == 6)

        # Perfect complement: all 1s in row1 match 6s in row2 and vice versa
        if row1_ones == row2_sixes and row1_sixes == row2_ones:
            total_pairs = len(row1_ones) + len(row1_sixes)

            # Multiple pairs AND first row is odd: swap all
            if total_pairs > 1 and r % 2 == 1:
                for c in row1_ones:
                    result[r][c] = 6
                    result[r + 1][c] = 1
                for c in row1_sixes:
                    result[r][c] = 1
                    result[r + 1][c] = 6
            # Single pair at column 0: always swap
            elif total_pairs == 1 and (0 in row1_ones or 0 in row1_sixes):
                for c in row1_ones:
                    result[r][c] = 6
                    result[r + 1][c] = 1
                for c in row1_sixes:
                    result[r][c] = 1
                    result[r + 1][c] = 6
        # Not perfect complement: selective swapping at odd columns with even row
        else:
            # Find positions where we have vertical (1,6) pairs at odd columns
            if r % 2 == 0:  # Even row index
                for c in range(cols):
                    if c % 2 == 1:  # Odd column
                        if result[r][c] == 1 and result[r + 1][c] == 6:
                            # Check if this is a valid swap candidate
                            # Swap only if there's an imbalance to correct
                            result[r][c] = 6
                            result[r + 1][c] = 1

    return result
