def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with background values (7s, 8s, or 9s) and scattered 1s at specific positions.
    2. Output transforms the grid by drawing cross-shaped patterns centered on each original 1.
    3. Each cross consists of vertical and horizontal lines of 1s extending through the entire grid.
    4. Original 1 positions become 2s, marking the intersection points of the crosses.
    5. 3s are placed diagonally adjacent to each 2, but only on cells that contain background values.
    6. The lines of 1s and intersection points of 2s remain unchanged when placing 3s.

    Procedure:
    1. Identify all positions containing 1s in the input grid.
    2. Create a working copy of the input grid to avoid modifying the original.
    3. For each original 1 position, draw complete vertical and horizontal lines of 1s.
    4. Replace each original 1 position with a 2 to mark the cross intersections.
    5. For each intersection point (2), check all four diagonal neighbors.
    6. Place 3s on diagonal neighbors that are within bounds and contain background values.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Create a copy

    # Find all positions of 1s
    ones_positions = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1:
                ones_positions.append((i, j))

    # Get the background value (most common value that's not 1)
    background_value = grid[0][0]  # Assume first cell is background

    # Step 1: Draw vertical and horizontal lines through each 1
    for row, col in ones_positions:
        # Draw vertical line
        for i in range(rows):
            result[i][col] = 1
        # Draw horizontal line
        for j in range(cols):
            result[row][j] = 1

    # Step 2: Replace original 1 positions with 2s
    for row, col in ones_positions:
        result[row][col] = 2

    # Step 3: Place 3s diagonally adjacent to each 2 (but not on lines of 1s)
    for row, col in ones_positions:
        # Check four diagonal positions
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonal neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (
                0 <= new_row < rows
                and 0 <= new_col < cols
                and result[new_row][new_col] != 1
                and result[new_row][new_col] != 2
            ):
                result[new_row][new_col] = 3

    return result
