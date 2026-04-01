def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10×width grid of all 0s except a 1 at position (9, 0)
    2. Output is a 10×width grid with a single 1 in each row creating a bouncing pattern
    3. The 1 bounces back and forth horizontally like a ball between walls
    4. Starting position depends on width: rightmost for even width, middle for odd width
    5. Direction initially moves left, reverses when hitting boundaries at 0 or width-1

    Procedure:
    1. Extract width from input grid dimensions
    2. Determine starting position: width-1 if even width, width//2 if odd width
    3. Initialize direction as -1 (moving left)
    4. For each of the 10 rows, place a 1 at current position
    5. After each row, update position by adding direction
    6. If next position hits boundary (< 0 or >= width), reverse direction and bounce
    """

    width = len(grid[0])
    height = len(grid)

    # Determine starting position and direction
    if width % 2 == 0:  # even width
        pos = width - 1
    else:  # odd width
        pos = width // 2

    direction = -1  # start moving left

    result = []

    for row in range(height):
        # Create output row with 1 at current position
        output_row = [0] * width
        output_row[pos] = 1
        result.append(output_row)

        # Move to next position
        next_pos = pos + direction

        # Check for bouncing off walls
        if next_pos < 0:
            direction = 1  # bounce right
            pos = 1
        elif next_pos >= width:
            direction = -1  # bounce left
            pos = width - 2
        else:
            pos = next_pos

    return result
