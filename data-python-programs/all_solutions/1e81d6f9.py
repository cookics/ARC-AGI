def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 15x15 grid of integers with 0s representing empty cells.
    2. Each grid has an L-shaped boundary formed by 5s at column 3 (rows 0-3) and row 3 (columns 0-3).
    3. This creates a top-left region (rows 0-3, columns 0-3).
    4. There's exactly one unique non-zero, non-5 number in this top-left region.
    5. Output is the same grid but with all instances of that number outside the top-left region removed (set to 0).

    Procedure:
    1. Identify the unique non-zero, non-5 number in the top-left region (rows 0-3, cols 0-3).
    2. Create output grid as copy of input grid.
    3. Remove all instances of that target number from positions outside top-left region by setting them to 0.
    """

    # Find the target number in top-left region (rows 0-3, columns 0-3)
    target_number = None
    for row in range(4):
        for col in range(4):
            value = grid[row][col]
            if value != 0 and value != 5:
                target_number = value
                break
        if target_number is not None:
            break

    # Create output grid as copy of input
    result = [row[:] for row in grid]

    # Remove target number from positions outside top-left region
    rows = len(grid)
    cols = len(grid[0])

    for row in range(rows):
        for col in range(cols):
            # If outside top-left region (row > 3 OR col > 3) and equals target number
            if (row > 3 or col > 3) and grid[row][col] == target_number:
                result[row][col] = 0

    return result
