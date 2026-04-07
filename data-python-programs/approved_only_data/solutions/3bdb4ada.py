def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing rectangular blocks of the same non-zero color surrounded by zeros.
    2. The output is a modified grid where rectangular blocks with exactly 3 rows have their middle row transformed.
    3. The transformation pattern creates alternating positions in the middle row between the original color and 0.
    4. The alternating pattern starts with the original color: color, 0, color, 0, color, ...

    Procedure:
    1. Create a copy of the input grid to store results.
    2. Find all rectangular blocks of non-zero colors by scanning the grid.
    3. For each discovered block, determine its exact rectangular bounds.
    4. Check if the block has exactly 3 rows in height.
    5. If it has 3 rows, modify the middle row to create an alternating pattern.
    6. In the middle row, set odd positions (relative to block start) to 0 while keeping even positions as original color.
    """

    # Create a copy of the grid
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find rectangular blocks
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                color = grid[r][c]

                # Find the rectangular bounds of this block
                min_r, max_r = r, r
                min_c, max_c = c, c

                # Expand to find the full rectangle
                # First, find the rightmost column with this color in this row
                while max_c + 1 < cols and grid[r][max_c + 1] == color:
                    max_c += 1

                # Then find the bottommost row where all columns have this color
                while max_r + 1 < rows:
                    valid_row = True
                    for col in range(min_c, max_c + 1):
                        if grid[max_r + 1][col] != color:
                            valid_row = False
                            break
                    if valid_row:
                        max_r += 1
                    else:
                        break

                # Mark all cells in this rectangle as visited
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        visited[rr][cc] = True

                # If this is a 3-row block, modify the middle row
                if max_r - min_r + 1 == 3:
                    middle_row = min_r + 1
                    for cc in range(min_c, max_c + 1):
                        # Create alternating pattern: start with original color
                        if (cc - min_c) % 2 == 1:  # Odd positions (1, 3, 5...) get 0
                            result[middle_row][cc] = 0

    return result
