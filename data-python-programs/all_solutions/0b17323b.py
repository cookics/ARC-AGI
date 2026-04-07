def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing 1s at specific positions forming a diagonal pattern.
    2. Output preserves all original 1s and adds 2s continuing the diagonal sequence.
    3. Example 1 has 1s at (0,0), (4,4), (8,8) with step +4 in both row and column.
    4. Output adds 2 at (12,12) continuing the same diagonal step pattern.
    5. Example 2 has 1s at (1,1), (3,3), (5,5) with step +2 in both row and column.
    6. Output adds 2s at (7,7), (9,9), (11,11), (13,13) continuing the diagonal sequence.
    7. The pattern is identifying diagonal arithmetic sequences of 1s and extending them with 2s.
    8. Extension continues until reaching grid boundaries.

    Procedure:
    1. Find all positions with value 1
    2. Determine if they form a diagonal arithmetic sequence
    3. Calculate the step size
    4. Continue the sequence with 2s until boundary
    5. Return grid with original 1s plus new 2s
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy input

    # Find all positions with value 1
    ones_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones_positions.append((r, c))

    if len(ones_positions) < 2:
        return result

    # Sort positions to identify the sequence
    ones_positions.sort()

    # Check if they form a diagonal arithmetic sequence
    # Calculate steps between consecutive positions
    step_r = ones_positions[1][0] - ones_positions[0][0]
    step_c = ones_positions[1][1] - ones_positions[0][1]

    # Verify it's a consistent diagonal sequence
    is_valid_sequence = True
    for i in range(2, len(ones_positions)):
        expected_r = ones_positions[i - 1][0] + step_r
        expected_c = ones_positions[i - 1][1] + step_c
        if ones_positions[i] != (expected_r, expected_c):
            is_valid_sequence = False
            break

    if not is_valid_sequence:
        return result

    # Continue the sequence with 2s
    last_r, last_c = ones_positions[-1]
    next_r = last_r + step_r
    next_c = last_c + step_c

    while 0 <= next_r < rows and 0 <= next_c < cols:
        result[next_r][next_c] = 2
        next_r += step_r
        next_c += step_c

    return result
