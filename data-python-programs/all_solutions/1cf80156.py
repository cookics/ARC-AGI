def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large sparse 2D grid (e.g., 12x12) containing mostly zeros with scattered non-zero elements.
    2. Non-zero elements can be different values (1, 2, 8, 6) representing different colors or objects.
    3. Output is a smaller rectangular sub-grid that forms the minimal bounding rectangle containing all non-zero elements.
    4. The output preserves the exact spatial relationships and values of elements within the bounding rectangle.
    5. Zero values within the bounding rectangle are preserved in the output.

    Procedure:
    1. Scan the entire grid to find all positions containing non-zero values.
    2. Calculate the minimal bounding box coordinates (min_row, max_row, min_col, max_col) that encompasses all non-zero positions.
    3. Extract the rectangular region defined by the bounding box coordinates.
    4. Return the extracted sub-grid as the result.
    """

    # Find all non-zero positions
    non_zero_positions = []
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] != 0:
                non_zero_positions.append((i, j))

    if not non_zero_positions:
        return []

    # Calculate bounding box
    min_row = min(pos[0] for pos in non_zero_positions)
    max_row = max(pos[0] for pos in non_zero_positions)
    min_col = min(pos[1] for pos in non_zero_positions)
    max_col = max(pos[1] for pos in non_zero_positions)

    # Extract the bounding rectangle
    result = []
    for i in range(min_row, max_row + 1):
        row = []
        for j in range(min_col, max_col + 1):
            row.append(grid[i][j])
        result.append(row)

    return result
