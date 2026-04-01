def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid is 13x14 with separator columns (all 0s) dividing it into 5 blocks of 2 columns each
    2. Last block is kept unchanged - it serves as reference or has special status
    3. For first N-1 blocks, use majority voting across those blocks only

    Procedure:
    1. Identify separator columns and blocks
    2. Keep last block unchanged
    3. For first N-1 blocks, use majority voting on odd rows
    """

    from collections import Counter

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if rows == 0:
        return grid

    result = [row[:] for row in grid]

    # Identify separator columns (all 0s)
    separators = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            separators.append(c)

    # Define blocks (2-column segments between separators)
    blocks = []
    prev = 0
    for sep in separators:
        if sep > prev:
            blocks.append((prev, sep))
        prev = sep + 1
    if prev < cols:
        blocks.append((prev, cols))

    # Only process 2-column blocks
    two_col_blocks = [b for b in blocks if b[1] - b[0] == 2]

    if len(two_col_blocks) < 2:
        return result

    # Exclude last block from corrections (keep it unchanged)
    blocks_to_correct = two_col_blocks[:-1]

    # For each odd row, use majority voting across ALL blocks (including last)
    for r in range(1, rows, 2):  # Odd rows only
        for col_offset in [0, 1]:  # First and second column of each block
            # Collect values from ALL blocks at this position (including last)
            values = []
            for c_start, c_end in two_col_blocks:  # Use all blocks for voting
                c = c_start + col_offset
                values.append(grid[r][c])

            # Find majority value
            if values:
                counter = Counter(values)
                majority_val, _ = counter.most_common(1)[0]

                # Apply majority to first N-1 blocks only
                for c_start, c_end in blocks_to_correct:
                    c = c_start + col_offset
                    result[r][c] = majority_val

    return result
