def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with mostly 0s and some non-zero values at specific positions.
    2. Output shows cross patterns where each non-zero value creates horizontal and vertical lines.
    3. For each non-zero value X at position (r,c), a horizontal line of X's is drawn across row r and a vertical line of X's down column c.
    4. At intersections where two different colored lines meet, the value 2 is placed.
    5. Original source positions retain their original non-zero values.
    6. Example: 8 at (1,3) creates horizontal line on row 1 and vertical line on column 3, 7 at (7,6) creates horizontal line on row 7 and vertical line on column 6.
    7. Intersections at (1,6) and (7,3) get value 2 because different colored lines meet there.

    Procedure:
    1. Find all non-zero positions and their values.
    2. Create result grid initialized with 0s.
    3. For each non-zero value, draw horizontal and vertical lines.
    4. Handle intersections by checking if different values meet.
    5. Place original values at their source positions.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all non-zero positions
    sources = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                sources.append((r, c, grid[r][c]))

    # Draw lines for each source
    for r, c, value in sources:
        # Draw horizontal line
        for col in range(cols):
            result[r][col] = value

        # Draw vertical line
        for row in range(rows):
            result[row][c] = value

    # Handle intersections of different lines
    for r in range(rows):
        for c in range(cols):
            # Check if this position is an intersection of different values
            horizontal_values = set()
            vertical_values = set()

            # Find which horizontal lines pass through this position
            for sr, sc, sv in sources:
                if sr == r:  # This source creates a horizontal line through (r,c)
                    horizontal_values.add(sv)
                if sc == c:  # This source creates a vertical line through (r,c)
                    vertical_values.add(sv)

            # If this is an intersection of different colored lines, place 2
            # But keep original source values
            is_source = any(sr == r and sc == c for sr, sc, sv in sources)
            if not is_source:
                # Check if horizontal and vertical lines have different values
                if horizontal_values and vertical_values:
                    if horizontal_values != vertical_values:
                        result[r][c] = 2

    return result
