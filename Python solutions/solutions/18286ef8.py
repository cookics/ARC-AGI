def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing integers, primarily filled with 7s and 0s forming a structure.
    2. Each grid contains exactly one 3x3 pattern of 5s with a 9 in the center.
    3. There is exactly one 6 somewhere else in the grid.
    4. The transformation replaces the 6 with 9 and moves the center 9 one step toward where the 6 was located.
    5. The original center position of the 9 gets replaced with 5.

    Procedure:
    1. Find the 3x3 pattern of 5s with 9 in center to locate the current 9 position.
    2. Find the position of the single 6 in the grid.
    3. Replace the 6 with 9 in the result grid.
    4. Calculate the direction from the center 9 to the 6's position.
    5. Move the center 9 one step toward the 6's original position.
    6. Replace the original center position with 5.
    """

    # Create a copy of the grid
    result = [row[:] for row in grid]

    # Find the center of the 3x3 pattern (where 9 is located)
    center_row, center_col = None, None
    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            if grid[i][j] == 9 and all(
                grid[i + di][j + dj] == 5
                for di in [-1, 0, 1]
                for dj in [-1, 0, 1]
                if (di, dj) != (0, 0)
            ):
                center_row, center_col = i, j
                break
        if center_row is not None:
            break

    # Find the position of the 6
    six_row, six_col = None, None
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 6:
                six_row, six_col = i, j
                break
        if six_row is not None:
            break

    # Replace the 6 with 9
    result[six_row][six_col] = 9

    # Calculate direction from center to 6
    direction_row = six_row - center_row
    direction_col = six_col - center_col

    # Determine movement: one step toward the 6
    # Take the sign of each component
    move_row = 1 if direction_row > 0 else (-1 if direction_row < 0 else 0)
    move_col = 1 if direction_col > 0 else (-1 if direction_col < 0 else 0)

    # If one direction is much stronger, prefer cardinal movement
    if abs(direction_col) >= 3 * abs(direction_row):
        move_row = 0
    elif abs(direction_row) >= 3 * abs(direction_col):
        move_col = 0

    # Move the 9 to the new position
    new_row = center_row + move_row
    new_col = center_col + move_col

    # Replace the center with 5 and put 9 at new position
    result[center_row][center_col] = 5
    result[new_row][new_col] = 9

    return result
