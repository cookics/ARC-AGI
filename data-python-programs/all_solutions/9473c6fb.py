def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid where most cells are 7 (background) and some cells have other values
    2. Output is the same grid with all non-7 values replaced by values from a cycle
    3. The cycle is [2, 8, 5] or [2, 5, 8] depending on count of non-7 values
    4. Non-7 values are processed in row-major order (left-to-right, top-to-bottom)
    5. If total non-7 values >= 9, use cycle [2, 5, 8], otherwise use [2, 8, 5]

    Procedure:
    1. Identify all non-7 positions in row-major order
    2. Count total non-7 values to determine which cycle to use
    3. Replace each non-7 value with the corresponding value from the cycle
    """

    # Create a copy of the grid
    result = [row[:] for row in grid]

    # Find all non-7 positions in row-major order
    non_seven_positions = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != 7:
                non_seven_positions.append((r, c))

    # Determine which cycle to use based on count
    total_count = len(non_seven_positions)
    if total_count >= 9:
        cycle = [2, 5, 8]
    else:
        cycle = [2, 8, 5]

    # Apply the transformation
    for i, (r, c) in enumerate(non_seven_positions):
        result[r][c] = cycle[i % len(cycle)]

    return result
