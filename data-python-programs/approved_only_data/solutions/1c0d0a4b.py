def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing 0s and 8s
    2. Output is a 2D grid containing only 0s and 2s (8s are completely removed)
    3. Pattern: positions become 2 when they are at intersections of rows and columns that contain 8s
    4. Only positions that are originally 0 can become 2 in the output
    5. The transformation identifies crossing points where 8-containing rows and columns meet at empty spaces

    Procedure:
    1. Initialize result grid with all zeros matching input dimensions
    2. Iterate through each position in the grid
    3. For each position that contains 0 in the input, check if it should become 2
    4. Mark position as 2 if both its row and column contain at least one 8
    5. Return the transformed result grid
    """

    def should_mark_as_2(grid, r, c):
        """
        Determine if position (r,c) should be marked as 2.
        Look at the surrounding pattern to decide.
        """
        rows = len(grid)
        cols = len(grid[0])

        # Check if this position has 8s in the same row AND column
        row_has_8 = any(grid[r][col] == 8 for col in range(cols) if col != c)
        col_has_8 = any(grid[row][c] == 8 for row in range(rows) if row != r)

        # Basic rule: if both row and column have 8s, consider marking as 2
        if row_has_8 and col_has_8:
            return True

        return False

    rows = len(grid)
    cols = len(grid[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:  # Only consider non-8 positions
                if should_mark_as_2(grid, r, c):
                    result[r][c] = 2

    return result
