def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 2x2 blocks of value 2 and scattered single cells of value 1
    2. Output moves the 1s to be adjacent to the 2x2 blocks
    3. For each block, the nearest 1 in each cardinal direction (up/down/left/right) moves adjacent to the block
    4. 1s that don't align with any block's rows or columns stay in place
    5. When multiple blocks compete for the same 1, the closest block (by Euclidean distance) wins

    Procedure:
    1. Find all 2×2 blocks of 2s in the grid
    2. For each block, scan in 4 directions to find the nearest 1 aligned with the block's rows/columns
    3. Track which 1s should move and where, resolving conflicts by distance to block center
    4. Apply all movements: remove 1s from old positions and place them in new positions
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all 2×2 blocks
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == 2 and grid[r][c+1] == 2 and
                grid[r+1][c] == 2 and grid[r+1][c+1] == 2):
                blocks.append((r, c))

    if not blocks:
        return result

    # Track which 1s should move and where
    moves = {}  # maps (r,c) -> (new_r, new_c)

    for block_r, block_c in blocks:
        # Look UP: scan column block_c and block_c+1, going upward from block_r-1
        closest_up = None
        for r in range(block_r - 1, -1, -1):
            if grid[r][block_c] == 1 or grid[r][block_c+1] == 1:
                c = block_c if grid[r][block_c] == 1 else block_c+1
                closest_up = (r, c)
                break

        # Look DOWN: scan column block_c and block_c+1, going downward from block_r+2
        closest_down = None
        for r in range(block_r + 2, rows):
            if grid[r][block_c] == 1 or grid[r][block_c+1] == 1:
                c = block_c if grid[r][block_c] == 1 else block_c+1
                closest_down = (r, c)
                break

        # Look LEFT: scan row block_r and block_r+1, going leftward from block_c-1
        closest_left = None
        for c in range(block_c - 1, -1, -1):
            if grid[block_r][c] == 1 or grid[block_r+1][c] == 1:
                r = block_r if grid[block_r][c] == 1 else block_r+1
                closest_left = (r, c)
                break

        # Look RIGHT: scan row block_r and block_r+1, going rightward from block_c+2
        closest_right = None
        for c in range(block_c + 2, cols):
            if grid[block_r][c] == 1 or grid[block_r+1][c] == 1:
                r = block_r if grid[block_r][c] == 1 else block_r+1
                closest_right = (r, c)
                break

        # Record movements based on Euclidean distance to block center
        MAX_DIST_TO_BLOCK = float('inf')

        if closest_up:
            r, c = closest_up
            new_pos = (block_r - 1, c)
            # Use Euclidean distance to block center for tie-breaking
            block_center_r = block_r + 0.5
            block_center_c = block_c + 0.5
            dist_to_block = ((r - block_center_r) ** 2 + (c - block_center_c) ** 2) ** 0.5
            if dist_to_block <= MAX_DIST_TO_BLOCK:
                if (r, c) not in moves:
                    moves[(r, c)] = (new_pos, dist_to_block)
                else:
                    old_dist = moves[(r, c)][1]
                    if dist_to_block < old_dist:
                        moves[(r, c)] = (new_pos, dist_to_block)

        if closest_down:
            r, c = closest_down
            new_pos = (block_r + 2, c)
            block_center_r = block_r + 0.5
            block_center_c = block_c + 0.5
            dist_to_block = ((r - block_center_r) ** 2 + (c - block_center_c) ** 2) ** 0.5
            if dist_to_block <= MAX_DIST_TO_BLOCK:
                if (r, c) not in moves:
                    moves[(r, c)] = (new_pos, dist_to_block)
                else:
                    old_dist = moves[(r, c)][1]
                    if dist_to_block < old_dist:
                        moves[(r, c)] = (new_pos, dist_to_block)

        if closest_left:
            r, c = closest_left
            new_pos = (r, block_c - 1)
            block_center_r = block_r + 0.5
            block_center_c = block_c + 0.5
            dist_to_block = ((r - block_center_r) ** 2 + (c - block_center_c) ** 2) ** 0.5
            if dist_to_block <= MAX_DIST_TO_BLOCK:
                if (r, c) not in moves:
                    moves[(r, c)] = (new_pos, dist_to_block)
                else:
                    old_dist = moves[(r, c)][1]
                    if dist_to_block < old_dist:
                        moves[(r, c)] = (new_pos, dist_to_block)

        if closest_right:
            r, c = closest_right
            new_pos = (r, block_c + 2)
            block_center_r = block_r + 0.5
            block_center_c = block_c + 0.5
            dist_to_block = ((r - block_center_r) ** 2 + (c - block_center_c) ** 2) ** 0.5
            if dist_to_block <= MAX_DIST_TO_BLOCK:
                if (r, c) not in moves:
                    moves[(r, c)] = (new_pos, dist_to_block)
                else:
                    old_dist = moves[(r, c)][1]
                    if dist_to_block < old_dist:
                        moves[(r, c)] = (new_pos, dist_to_block)

    # Apply movements
    # First, clear all 1s that will move
    for (old_r, old_c) in moves.keys():
        result[old_r][old_c] = 0

    # Place them in new positions
    for (old_r, old_c), (new_pos, dist) in moves.items():
        new_r, new_c = new_pos
        result[new_r][new_c] = 1

    return result
