def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing 0s and 2s
    2. Output fills certain rows with 2s based on pattern matching rules
    3. If any row has horizontally adjacent 2s, the grid remains unchanged
    4. Otherwise, rows that are identical to another row AND have exactly two 2s get filled
    5. Filling means setting all values between the two 2s (inclusive) to 2

    Procedure:
    1. Check if there are any horizontally adjacent 2s anywhere in the grid
    2. If adjacent 2s exist, return original grid unchanged
    3. If no adjacent 2s, find rows that are identical to another row and have exactly two 2s
    4. Fill positions between the 2s in those matching rows only
    """

    # Check for horizontally adjacent 2's anywhere in the grid
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols - 1):
            if grid[r][c] == 2 and grid[r][c + 1] == 2:
                # Found adjacent 2's, return grid unchanged
                return [row[:] for row in grid]

    # No adjacent 2's found, find rows to fill
    result = [row[:] for row in grid]

    # Find pairs of identical rows with exactly two 2's
    identical_pairs = []

    for i in range(len(grid)):
        row = grid[i]
        twos_positions = [j for j, val in enumerate(row) if val == 2]

        # Check if this row has exactly two 2's
        if len(twos_positions) == 2:
            # Check for identical rows
            for j in range(i + 1, len(grid)):
                if grid[j] == row:
                    identical_pairs.append((i, j))

    # If multiple pairs exist, prioritize consecutive rows
    rows_to_fill = set()

    if identical_pairs:
        # Check for consecutive pairs first
        consecutive_pairs = [(i, j) for i, j in identical_pairs if j == i + 1]

        if consecutive_pairs:
            # Fill all consecutive pairs
            for i, j in consecutive_pairs:
                rows_to_fill.add(i)
                rows_to_fill.add(j)
        else:
            # No consecutive pairs, fill all pairs
            for i, j in identical_pairs:
                rows_to_fill.add(i)
                rows_to_fill.add(j)

    # Fill the identified rows
    for row_idx in rows_to_fill:
        row = result[row_idx]
        twos_positions = [i for i, val in enumerate(row) if val == 2]
        if len(twos_positions) == 2:
            start, end = twos_positions[0], twos_positions[1]
            for i in range(start, end + 1):
                result[row_idx][i] = 2

    return result
