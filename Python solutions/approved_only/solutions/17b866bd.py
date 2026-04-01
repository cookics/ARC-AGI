def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing primarily values 0 and 8 in a regular pattern, with occasional special values (1, 4, 7, etc.).
    2. The grid follows a repeating 5x5 block structure where 0s appear at block corners (positions where both row % 5 == 0 and col % 5 == 0).
    3. All other boundary positions in the 5x5 blocks contain 8s, forming a border around each block.
    4. Special values (not 0 or 8) can appear at any position and indicate which blocks need transformation.
    5. The output preserves the original structure but fills the 3x3 interior of affected blocks with the corresponding special value.
    6. After transformation, all special value positions are restored to their expected boundary values (0 for corners, 8 for edges).

    Procedure:
    1. Scan the entire grid to identify all special values (any value that is not 0 or 8) and their positions.
    2. Also detect anomalous boundary values (8s where 0s should be, or 0s where 8s should be) as these also trigger transformations.
    3. For each special position found, determine which 5x5 block it belongs to by calculating block coordinates.
    4. Fill the 3x3 interior of each affected block with the special value found in that block.
    5. The filling pattern covers positions (1,1) to (3,3) relative to the block's top-left corner, creating a cross-like shape.
    6. Restore all originally special positions to their correct boundary values based on the 5x5 block pattern.
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all special values (not 0 or 8) and anomalous patterns
    special_positions = []
    for r in range(rows):
        for c in range(cols):
            expected_val = 0 if (r % 5 == 0 and c % 5 == 0) else 8

            if grid[r][c] not in [0, 8]:
                special_positions.append((r, c, grid[r][c]))
            elif r % 5 == 0 and c % 5 == 0 and grid[r][c] != 0:
                # Anomalous 8 where 0 should be
                special_positions.append((r, c, 8))
            elif r % 5 == 0 and c % 5 != 0 and grid[r][c] != 8:
                # Anomalous 0 where 8 should be
                special_positions.append((r, c, 8))

    # Process each special position
    for r, c, value in special_positions:
        # Determine which 5x5 block this position is in
        block_r = r // 5
        block_c = c // 5

        # Base position of the 5x5 block
        base_r = block_r * 5
        base_c = block_c * 5

        # Apply the specific pattern within the 5x5 block
        # Looking at the examples, the pattern fills the 3x3 interior in a cross shape
        if base_r + 1 < rows and base_c + 2 < cols:
            result[base_r + 1][base_c + 2] = value
        if base_r + 1 < rows and base_c + 3 < cols:
            result[base_r + 1][base_c + 3] = value

        if base_r + 2 < rows and base_c + 1 < cols:
            result[base_r + 2][base_c + 1] = value
        if base_r + 2 < rows and base_c + 2 < cols:
            result[base_r + 2][base_c + 2] = value
        if base_r + 2 < rows and base_c + 3 < cols:
            result[base_r + 2][base_c + 3] = value
        if base_r + 2 < rows and base_c + 4 < cols:
            result[base_r + 2][base_c + 4] = value

        if base_r + 3 < rows and base_c + 1 < cols:
            result[base_r + 3][base_c + 1] = value
        if base_r + 3 < rows and base_c + 2 < cols:
            result[base_r + 3][base_c + 2] = value
        if base_r + 3 < rows and base_c + 3 < cols:
            result[base_r + 3][base_c + 3] = value
        if base_r + 3 < rows and base_c + 4 < cols:
            result[base_r + 3][base_c + 4] = value

        if base_r + 4 < rows and base_c + 2 < cols:
            result[base_r + 4][base_c + 2] = value
        if base_r + 4 < rows and base_c + 3 < cols:
            result[base_r + 4][base_c + 3] = value

    # Restore all special positions to their normal boundary values
    for r, c, value in special_positions:
        if r % 5 == 0 and c % 5 == 0:
            result[r][c] = 0  # Corner positions
        else:
            result[r][c] = 8  # Other boundary positions

    return result
