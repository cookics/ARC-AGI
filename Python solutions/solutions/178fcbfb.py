def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing values 0, 1, 2, and 3.
    2. Output is a 2D grid where special values trigger fill operations.
    3. When a cell contains value 1 or 3, fill the entire row with that value.
    4. When a cell contains value 2, fill the entire column with that value.
    5. When there's overlap, row operations (1s and 3s) take precedence over column operations (2s).

    Procedure:
    1. Create a copy of the input grid.
    2. Find all positions with value 2 and fill their columns with 2s.
    3. Find all positions with value 1 or 3 and fill their rows with those values (overriding any previous column fills).
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # First pass: handle 2s (column fills)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                # Fill entire column with 2s
                for row_idx in range(rows):
                    result[row_idx][c] = 2

    # Second pass: handle 1s and 3s (row fills) - these take precedence
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 or grid[r][c] == 3:
                # Fill entire row with this value
                for col_idx in range(cols):
                    result[r][col_idx] = grid[r][c]

    return result
