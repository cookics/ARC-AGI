def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing values 0, 2, and 8.
    2. Each grid has exactly two 2x2 blocks composed entirely of 2s.
    3. The output is always a single-element list containing either [[0]] or [[8]].
    4. The pattern depends on the spatial relationship between the two 2x2 blocks.
    5. If blocks overlap in rows or columns, output is [[8]].
    6. If blocks don't overlap, the slope between their centers determines the output.

    Procedure:
    1. Scan the grid to locate all 2x2 blocks of 2s by checking each position.
    2. Sort the found blocks by position for consistent ordering.
    3. Calculate whether the blocks overlap in rows or columns.
    4. If blocks overlap in rows or columns, return [[8]].
    5. If blocks don't overlap, calculate the slope between their center points.
    6. Return [[8]] if slope equals 1, otherwise return [[0]].
    """

    rows, cols = len(grid), len(grid[0])

    # Find 2x2 blocks of 2s
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (
                grid[r][c] == 2
                and grid[r][c + 1] == 2
                and grid[r + 1][c] == 2
                and grid[r + 1][c + 1] == 2
            ):
                blocks.append((r, c))

    # Should have exactly 2 blocks
    assert len(blocks) == 2

    # Sort blocks by row, then by column for consistent ordering
    blocks.sort()
    block1, block2 = blocks
    r1, c1 = block1
    r2, c2 = block2

    # Check if the blocks share a row or column
    # If block1 rows are r1,r1+1 and block2 rows are r2,r2+1
    # They share a row if the ranges overlap
    rows_overlap = not (r1 + 1 < r2 or r2 + 1 < r1)
    cols_overlap = not (c1 + 1 < c2 or c2 + 1 < c1)

    # Pattern analysis:
    # If they overlap in rows or columns -> 8
    # If they don't overlap, check if the line connecting centers has positive slope
    if rows_overlap or cols_overlap:
        return [[8]]
    else:
        # Centers of the blocks
        center1 = (r1 + 0.5, c1 + 0.5)
        center2 = (r2 + 0.5, c2 + 0.5)

        # Calculate slope
        # If the slope is exactly 1, return 8
        # Otherwise, return 0
        delta_y = center2[0] - center1[0]
        delta_x = center2[1] - center1[1]

        if delta_x != 0 and abs(delta_y / delta_x - 1.0) < 1e-9:
            return [[8]]
        else:
            return [[0]]
