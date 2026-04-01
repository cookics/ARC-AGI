def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 13x13 grid filled with zeros and exactly two "2"s positioned as diagonal corners.
    2. Output creates a rectangular structure using the two "2"s as corner references.
    3. The rectangle's horizontal borders extend across the entire grid width forming full horizontal lines.
    4. The rectangle's vertical borders extend from top to bottom of the grid forming full vertical lines.
    5. The interior of the rectangle is filled with "1"s.
    6. All structural lines and borders are drawn with "2"s.

    Procedure:
    1. Find the positions of the two "2"s in the input grid.
    2. Determine the rectangle boundaries using min/max row and column coordinates.
    3. Draw horizontal borders across the entire grid width at the top and bottom rows of the rectangle.
    4. Draw vertical borders from top to bottom of the grid at the left and right columns of the rectangle.
    5. Fill the interior rectangle area with "1"s.
    """

    # Create a copy of the input grid
    result = [row[:] for row in grid]

    # Find the two "2" positions
    twos = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                twos.append((i, j))

    # Get rectangle boundaries
    r1, c1 = twos[0]
    r2, c2 = twos[1]

    min_row = min(r1, r2)
    max_row = max(r1, r2)
    min_col = min(c1, c2)
    max_col = max(c1, c2)

    # Draw horizontal borders of rectangle across entire grid width
    for col in range(len(grid[0])):
        result[min_row][col] = 2
        result[max_row][col] = 2

    # Draw vertical borders of rectangle
    for row in range(min_row, max_row + 1):
        result[row][min_col] = 2
        result[row][max_col] = 2

    # Fill interior with 1s
    for row in range(min_row + 1, max_row):
        for col in range(min_col + 1, max_col):
            result[row][col] = 1

    # Extend vertical lines from left and right edges to grid boundaries
    for row in range(len(grid)):
        result[row][min_col] = 2
        result[row][max_col] = 2

    return result
