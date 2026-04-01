def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains cells with value 5 forming a shape
    2. Output replaces 5s with either 8 (forming 2×2 blocks) or 2 (individual cells)
    3. The 2×2 blocks follow a checkerboard tiling pattern
    4. Grid is tiled with 2×2 blocks starting at even row/col positions
    5. Blocks are selected based on checkerboard parity: (row//2 + col//2) % 2
    6. The parity is determined by the first valid 2×2 block of 5s found

    Procedure:
    1. Tile the grid into 2×2 blocks at even positions (0,0), (0,2), (0,4), etc.
    2. Find all 2×2 blocks that are completely filled with 5s
    3. Determine the checkerboard parity from the first valid block
    4. Select all blocks with matching parity
    5. Mark selected blocks as 8, remaining 5s as 2
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find ALL possible 2×2 blocks that are completely filled with 5s
    # Check every possible starting position
    # Prefer blocks where columns don't extend downward
    valid_blocks = []

    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check if this 2×2 block is completely filled with 5s
            if (grid[r][c] == 5 and grid[r][c+1] == 5 and
                grid[r+1][c] == 5 and grid[r+1][c+1] == 5):
                # Check if BOTH columns extend downward beyond the block
                extends_down = False
                if r + 2 < rows:
                    if grid[r+2][c] == 5 and grid[r+2][c+1] == 5:
                        extends_down = True

                # Compute checkerboard parity based on the block's position
                block_idx_r = r // 2
                block_idx_c = c // 2
                parity = (block_idx_r + block_idx_c) % 2
                valid_blocks.append((r, c, parity, extends_down))

    # Determine the checkerboard parity from the first valid block that doesn't extend down
    # If no such blocks, use parity from first block
    target_parity = None
    for r, c, p, ext in valid_blocks:
        if not ext:
            target_parity = p
            break
    if target_parity is None and valid_blocks:
        target_parity = valid_blocks[0][2]
    if target_parity is None:
        target_parity = 0

    # Select non-overlapping blocks with matching parity
    # Prefer blocks that don't extend downward
    # Process blocks from top-left to bottom-right
    result = [row[:] for row in grid]
    assigned = [[False] * cols for _ in range(rows)]

    # Group blocks by their overlap and select the best one from each group
    # A greedy approach: process blocks and skip overlapping ones
    # Process from top-left to bottom-right, but within same row prefer right to left
    valid_blocks_filtered = [b for b in valid_blocks if b[2] == target_parity]

    # Sort: by row (asc), then by col (desc), then by non-extending
    valid_blocks_sorted = sorted(valid_blocks_filtered,
                                 key=lambda x: (x[0], -x[1], x[3]))

    for r, c, parity, extends in valid_blocks_sorted:
        # Check if any cell in this block is already assigned
        cells = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
        if not any(assigned[rr][cc] for rr, cc in cells):
            # Mark this 2×2 block as 8
            for rr, cc in cells:
                result[rr][cc] = 8
                assigned[rr][cc] = True

    # Mark remaining 5s as 2
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not assigned[r][c]:
                result[r][c] = 2

    return result
