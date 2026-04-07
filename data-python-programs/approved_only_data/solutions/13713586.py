def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 0s (empty cells), 5s (boundary walls), and colored numbers (2,3,4,6,7,8)
    2. Output is the same grid with colored segments expanded toward boundary walls
    3. The 5s form boundary walls along one edge of the grid (top, bottom, left, or right)
    4. Colored segments expand in the direction toward these boundary walls
    5. When multiple segments exist in the same row/column, they partition the space between them

    Procedure:
    1. Identify which edge has the boundary walls (top/bottom/left/right edge of 5s)
    2. For each row/column perpendicular to the boundary, find all colored segments
    3. Expand each segment toward the boundary wall
    4. When multiple segments exist in the same line, partition the space between them
    5. Return the modified grid with expanded segments
    """

    height, width = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Identify boundary direction
    right_boundary = (
        all(grid[r][-1] == 5 for r in range(height)) if width > 0 else False
    )
    left_boundary = all(grid[r][0] == 5 for r in range(height)) if width > 0 else False
    bottom_boundary = (
        all(grid[-1][c] == 5 for c in range(width)) if height > 0 else False
    )
    top_boundary = all(grid[0][c] == 5 for c in range(width)) if height > 0 else False

    if right_boundary:
        # Expand horizontally toward right boundary
        for r in range(height):
            # Find all colored cells in this row
            colored_cells = []
            for c in range(width):
                if grid[r][c] not in [0, 5]:
                    colored_cells.append((c, grid[r][c]))

            if colored_cells:
                colored_cells.sort()  # sort by column position

                # Expand each segment toward the boundary
                for i, (col, color) in enumerate(colored_cells):
                    # Determine end position
                    if i + 1 < len(colored_cells):
                        # Next segment exists, expand until just before it
                        end_col = colored_cells[i + 1][0]
                    else:
                        # No next segment, expand to boundary
                        end_col = width - 1

                    # Fill from current position to end
                    for c in range(col, end_col):
                        result[r][c] = color

    elif left_boundary:
        # Expand horizontally toward left boundary
        for r in range(height):
            colored_cells = []
            for c in range(width):
                if grid[r][c] not in [0, 5]:
                    colored_cells.append((c, grid[r][c]))

            if colored_cells:
                colored_cells.sort(reverse=True)  # sort in descending order

                for i, (col, color) in enumerate(colored_cells):
                    # Determine start position
                    if i + 1 < len(colored_cells):
                        start_col = colored_cells[i + 1][0] + 1
                    else:
                        start_col = 1  # after boundary

                    for c in range(start_col, col + 1):
                        result[r][c] = color

    elif bottom_boundary:
        # Expand vertically toward bottom boundary
        for c in range(width):
            colored_cells = []
            for r in range(height):
                if grid[r][c] not in [0, 5]:
                    colored_cells.append((r, grid[r][c]))

            if colored_cells:
                colored_cells.sort()  # sort by row position

                for i, (row, color) in enumerate(colored_cells):
                    if i + 1 < len(colored_cells):
                        end_row = colored_cells[i + 1][0]
                    else:
                        end_row = height - 1

                    for r in range(row, end_row):
                        result[r][c] = color

    elif top_boundary:
        # Expand vertically toward top boundary
        for c in range(width):
            colored_cells = []
            for r in range(height):
                if grid[r][c] not in [0, 5]:
                    colored_cells.append((r, grid[r][c]))

            if colored_cells:
                colored_cells.sort(reverse=True)  # sort in descending order

                for i, (row, color) in enumerate(colored_cells):
                    if i + 1 < len(colored_cells):
                        start_row = colored_cells[i + 1][0] + 1
                    else:
                        start_row = 1  # after boundary

                    for r in range(start_row, row + 1):
                        result[r][c] = color

    return result
