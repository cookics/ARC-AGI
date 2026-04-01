def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing values 0, 1, and 8.
    2. The 1s form specific geometric patterns in one region of the grid.
    3. The 8s form patterns in a separate region of the grid.
    4. The output transforms all 8s to a specific replacement value based on the 1s pattern.
    5. All 1s are removed (become 0s) in the output.
    6. Three distinct 1s patterns exist: T-like with top maps to 7, T-like with bottom maps to 3, cross/plus maps to 2.

    Procedure:
    1. Find all positions containing 1s in the grid.
    2. Extract the bounding box of the 1s pattern.
    3. Normalize the pattern to a relative coordinate system.
    4. Match the pattern against three known templates.
    5. Replace all 8s with the corresponding replacement value.
    6. Replace all 1s with 0s to create the output grid.
    """

    # Find all 1s positions
    ones_positions = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                ones_positions.append((r, c))

    if not ones_positions:
        # No 1s found, return grid as is
        return [row[:] for row in grid]

    # Find bounding box of 1s
    min_r = min(pos[0] for pos in ones_positions)
    max_r = max(pos[0] for pos in ones_positions)
    min_c = min(pos[1] for pos in ones_positions)
    max_c = max(pos[1] for pos in ones_positions)

    # Extract the pattern relative to bounding box
    pattern = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            if grid[r][c] == 1:
                row.append(1)
            else:
                row.append(0)
        pattern.append(row)

    # Define the three known patterns
    pattern_a = [[1, 1, 1], [1, 0, 1], [0, 1, 0]]  # → 7
    pattern_b = [[1, 0, 1], [0, 1, 0], [1, 1, 1]]  # → 3
    pattern_c = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # → 2

    # Determine replacement value
    replacement_value = 0
    if pattern == pattern_a:
        replacement_value = 7
    elif pattern == pattern_b:
        replacement_value = 3
    elif pattern == pattern_c:
        replacement_value = 2
    else:
        # Default case - shouldn't happen based on training data
        replacement_value = 1

    # Create output grid
    result = []
    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            if grid[r][c] == 8:
                row.append(replacement_value)
            elif grid[r][c] == 1:
                row.append(0)
            else:
                row.append(grid[r][c])
        result.append(row)

    return result
