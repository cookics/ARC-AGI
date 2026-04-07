def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3x3 grid
    2. Output is a 9x9 grid filled with zeros except for a tiled region
    3. If any row has all same elements, tile the 3x3 pattern horizontally 3 times
    4. If row 1 (middle) is uniform, place tiled pattern at rows 3-5
    5. If row 0 or 2 is uniform, place tiled pattern at rows 0-2
    6. If no row is uniform, tile the 3x3 pattern vertically 3 times at columns 0-2

    Procedure:
    1. Create 9x9 output grid filled with zeros
    2. Check each row to find if any has all elements the same
    3. If uniform row found, determine starting row (0 or 3) based on row index
    4. Tile the 3x3 pattern horizontally 3 times in the target row range
    5. If no uniform row found, tile the 3x3 pattern vertically 3 times in columns 0-2
    """

    # Create 9x9 output grid filled with zeros
    result = [[0 for _ in range(9)] for _ in range(9)]

    # Check for rows with all same elements
    all_same_row_index = None
    for i, row in enumerate(grid):
        if len(set(row)) == 1:  # All elements in row are the same
            all_same_row_index = i
            break

    if all_same_row_index is not None:
        # Horizontal repetition
        if all_same_row_index == 1:  # Middle row has all same values
            # Place pattern at rows 3-5
            start_row = 3
        else:
            # Place pattern at rows 0-2
            start_row = 0

        for i in range(3):
            for j in range(9):
                result[start_row + i][j] = grid[i][j % 3]
    else:
        # Vertical repetition - repeat the entire 3x3 pattern 3 times vertically
        for i in range(9):
            for j in range(3):
                result[i][j] = grid[i % 3][j]

    return result
