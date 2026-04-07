def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an 11x11 grid containing 0s (empty), 5s (separator lines), and one type of colored value.
    2. Output is an 11x11 grid where quadrant(s) with the most colored cells are completely filled, others cleared to 0s.
    3. The grid is divided by lines of 5s at rows 3,7 and columns 3,7 into 9 quadrants of size 3x3.
    4. Pattern: Find quadrant(s) with maximum count of colored cells and fill them entirely with that color.
    5. Separator lines of 5s always remain unchanged in the output.

    Procedure:
    1. Create a copy of the input grid
    2. Identify the colored value (non-0, non-5 value)
    3. For each of the 9 quadrants, count how many cells contain the colored value
    4. Find the maximum count
    5. For quadrants with maximum count, fill entirely with the colored value
    6. For all other quadrants, clear to 0s
    7. Keep separator lines (rows 3,7 and cols 3,7) as 5s
    """

    result = [row[:] for row in grid]  # Deep copy

    # Find the colored value (non-0, non-5)
    colored_value = None
    for row in grid:
        for cell in row:
            if cell != 0 and cell != 5:
                colored_value = cell
                break
        if colored_value is not None:
            break

    if colored_value is None:
        return result

    # Define quadrant boundaries
    quadrant_rows = [(0, 3), (4, 7), (8, 11)]
    quadrant_cols = [(0, 3), (4, 7), (8, 11)]

    # Count colored cells in each quadrant
    quadrant_counts = []
    for i, (r_start, r_end) in enumerate(quadrant_rows):
        for j, (c_start, c_end) in enumerate(quadrant_cols):
            count = 0
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if grid[r][c] == colored_value:
                        count += 1
            quadrant_counts.append((count, i, j))

    # Find maximum count
    max_count = max(count for count, _, _ in quadrant_counts)

    # Clear all quadrants to 0s first
    for i, (r_start, r_end) in enumerate(quadrant_rows):
        for j, (c_start, c_end) in enumerate(quadrant_cols):
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    result[r][c] = 0

    # Fill quadrants with maximum count
    for count, i, j in quadrant_counts:
        if count == max_count:
            r_start, r_end = quadrant_rows[i]
            c_start, c_end = quadrant_cols[j]
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    result[r][c] = colored_value

    # Restore separator lines
    for r in [3, 7]:
        for c in range(11):
            result[r][c] = 5
    for c in [3, 7]:
        for r in range(11):
            result[r][c] = 5

    return result
