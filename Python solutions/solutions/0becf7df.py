def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid of integers, output is a 2D grid of the same size with some values swapped
    2. The top-left 2x2 region acts as an "anchor" defining value pairs that determine swapping rules
    3. The anchor region defines swap pairs: first row values swap with each other, second row values swap with each other
    4. Throughout the rest of the grid, values are swapped according to the pairs defined by the anchor
    5. The anchor region itself remains unchanged in the output

    Procedure:
    1. Extract the 2x2 anchor pattern from the top-left corner of the input grid
    2. Create a swap mapping from the anchor values (first row pair and second row pair)
    3. Apply the swapping transformation to all positions except the anchor region
    4. Keep the anchor region unchanged and return the transformed grid
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy input

    # Extract anchor pattern from top-left 2x2
    anchor = [[grid[0][0], grid[0][1]], [grid[1][0], grid[1][1]]]

    # Create swap mapping from anchor values
    # anchor[0][0] ↔ anchor[0][1]
    # anchor[1][0] ↔ anchor[1][1]
    swap_map = {}

    # First pair: top row of anchor
    val1, val2 = anchor[0][0], anchor[0][1]
    if val1 != 0 and val2 != 0:
        swap_map[val1] = val2
        swap_map[val2] = val1

    # Second pair: bottom row of anchor
    val3, val4 = anchor[1][0], anchor[1][1]
    if val3 != 0 and val4 != 0:
        swap_map[val3] = val4
        swap_map[val4] = val3

    # Apply swapping to all positions except anchor region
    for r in range(rows):
        for c in range(cols):
            # Skip anchor region (top-left 2x2)
            if r < 2 and c < 2:
                continue

            original_val = grid[r][c]
            if original_val in swap_map:
                result[r][c] = swap_map[original_val]
            # If value not in swap map, keep unchanged

    return result
