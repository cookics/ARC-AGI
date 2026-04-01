def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains values 0, 1, 2, and 3
    2. The value 0 appears in rectangular blocks (holes to be filled)
    3. The value 2 appears in rectangular blocks (markers)
    4. Pattern analysis:
       - Examples 1, 4: 2s at row 0 or last row → swap (2s→0s, 0s→1s)
       - Examples 2, 3: 2s elsewhere → keep 2s, move 0s to new location
       - New location: row = midpoint of (0s row, 2s row), col = 2s col + 7

    Procedure:
    1. Find all 0s and 2s positions
    2. Always fill 0s with 1
    3. Check if 2s should become 0s (based on row position)
    4. If 2s stay as 2s, calculate new position for 0s block
    """
    # Create a deep copy of the grid
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all positions with 0s and 2s
    zeros = []
    twos = []

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                zeros.append((i, j))
            elif grid[i][j] == 2:
                twos.append((i, j))

    if not zeros or not twos:
        return result

    # Calculate bounding boxes
    zero_rows = [r for r, c in zeros]
    zero_cols = [c for r, c in zeros]
    two_rows = [r for r, c in twos]
    two_cols = [c for r, c in twos]

    zero_r_min, zero_r_max = min(zero_rows), max(zero_rows)
    zero_c_min, zero_c_max = min(zero_cols), max(zero_cols)
    two_r_min, two_r_max = min(two_rows), max(two_rows)
    two_c_min, two_c_max = min(two_cols), max(two_cols)

    zero_r_center = (zero_r_min + zero_r_max) / 2
    zero_c_center = (zero_c_min + zero_c_max) / 2
    two_r_center = (two_r_min + two_r_max) / 2
    two_c_center = (two_c_min + two_c_max) / 2

    block_height = zero_r_max - zero_r_min + 1
    block_width = zero_c_max - zero_c_min + 1

    # Replace all 0s with 1
    for i, j in zeros:
        result[i][j] = 1

    # Check if 2s should become 0s (if top-left corner at row 0 or last row)
    if two_r_min == 0 or two_r_min == rows - 1:
        # Simple swap: 2s become 0s
        for i, j in twos:
            result[i][j] = 0
    else:
        # Keep 2s, place 0s at new location
        new_r_center = (zero_r_center + two_r_center) / 2

        # Offset depends on which side the 2s are on
        if two_c_center < cols / 2:
            # 2s on left side, place 0s to the right
            new_c_center = two_c_center + 7
        else:
            # 2s on right side, place 0s to the left
            new_c_center = two_c_center - 7

        # Calculate new block position
        new_r_min = int(new_r_center - block_height / 2 + 0.5)
        new_c_min = int(new_c_center - block_width / 2 + 0.5)

        # Place 0s at new location
        for i in range(block_height):
            for j in range(block_width):
                result[new_r_min + i][new_c_min + j] = 0

    return result
