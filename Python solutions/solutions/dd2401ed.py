def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a vertical line of 5s (separator)
    2. Output moves the separator to a new column
    3. The new column position = old_column * 2 + 1
    4. If new position > 7 (midpoint), 2s between old and new positions become 1s
    5. All other values and positions remain unchanged

    Procedure:
    1. Find the column containing all 5s (separator column)
    2. Calculate new separator position: new_col = old_col * 2 + 1
    3. Copy the grid
    4. Move the separator from old to new position
    5. If new_col > 7, convert 2s between old_col and new_col to 1s
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find the column with 5s (separator)
    old_col = -1
    for col in range(cols):
        if all(grid[row][col] == 5 for row in range(rows)):
            old_col = col
            break

    # Calculate new separator position
    new_col = old_col * 2 + 1

    # Move the separator
    for row in range(rows):
        result[row][old_col] = 0  # Clear old position
        result[row][new_col] = 5  # Set new position

    # Convert 2s to 1s if separator moves past midpoint
    midpoint = cols // 2
    if new_col > midpoint:
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 2 and old_col < col < new_col:
                    result[row][col] = 1

    return result
