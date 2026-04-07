def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with value 8 forming cross-like structures (horizontal and vertical lines).
    2. These 8s divide the grid into rectangular regions.
    3. Output fills specific interior regions with different colored values.
    4. Top-middle region gets filled with value 2.
    5. Middle-left region gets filled with value 4.
    6. Middle-center region gets filled with value 6.
    7. Middle-right region gets filled with value 3.
    8. Bottom-middle region gets filled with value 1.
    9. Edge regions (touching grid boundaries) remain 0.
    10. The 8s themselves remain unchanged.

    Procedure:
    1. Find horizontal lines (rows completely filled with 8s).
    2. Find vertical lines (columns completely filled with 8s).
    3. Identify the regions created by these lines.
    4. Fill the interior regions with appropriate colors based on position.
    """

    # Make a copy of the grid
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find horizontal lines of 8s
    horizontal_lines = []
    for r in range(rows):
        if all(grid[r][c] == 8 for c in range(cols)):
            horizontal_lines.append(r)

    # Find vertical lines of 8s
    vertical_lines = []
    for c in range(cols):
        if all(grid[r][c] == 8 for r in range(rows)):
            vertical_lines.append(c)

    # Sort the lines
    horizontal_lines.sort()
    vertical_lines.sort()

    # Create regions and fill them
    # We need at least 2 horizontal and 2 vertical lines to create interior regions
    if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
        h1, h2 = horizontal_lines[0], horizontal_lines[1]
        v1, v2 = vertical_lines[0], vertical_lines[1]

        # Fill regions based on position
        # Top-middle region (2)
        for r in range(0, h1):
            for c in range(v1 + 1, v2):
                if result[r][c] != 8:
                    result[r][c] = 2

        # Middle-left region (4)
        for r in range(h1 + 1, h2):
            for c in range(0, v1):
                if result[r][c] != 8:
                    result[r][c] = 4

        # Middle-center region (6)
        for r in range(h1 + 1, h2):
            for c in range(v1 + 1, v2):
                if result[r][c] != 8:
                    result[r][c] = 6

        # Middle-right region (3)
        for r in range(h1 + 1, h2):
            for c in range(v2 + 1, cols):
                if result[r][c] != 8:
                    result[r][c] = 3

        # Bottom-middle region (1)
        for r in range(h2 + 1, rows):
            for c in range(v1 + 1, v2):
                if result[r][c] != 8:
                    result[r][c] = 1

    return result
