def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 9x9 grid with pattern in one 3x3 block
    2. Output replicates pattern to multiple blocks based on source position
    3. Block layout (0-indexed):
       (0,0) (0,1) (0,2)
       (1,0) (1,1) (1,2)
       (2,0) (2,1) (2,2)
    4. Pattern rules based on source position type:
       - Center (r=1, c=1): fills anti-diagonal blocks (0,2), (1,1), (2,0) + (1,2)
       - Edge (r=1 XOR c=1): fills all 4 edge blocks + (2,0)
       - Corner (r≠1 AND c≠1): fills source + same-row neighbor + (1,2) + (2,0)

    Procedure:
    1. Find source block containing the pattern
    2. Extract pattern as relative coordinates
    3. Determine target blocks using position-based rules
    4. Copy pattern to all target blocks
    """

    result = [row[:] for row in grid]

    # Find source block
    pattern_block = None
    pattern = {}

    for block_r in range(3):
        for block_c in range(3):
            has_pattern = False
            for r in range(block_r * 3, (block_r + 1) * 3):
                for c in range(block_c * 3, (block_c + 1) * 3):
                    if grid[r][c] != 0:
                        has_pattern = True
                        pattern[(r - block_r * 3, c - block_c * 3)] = grid[r][c]

            if has_pattern:
                pattern_block = (block_r, block_c)
                break
        if pattern_block:
            break

    if not pattern_block:
        return result

    r, c = pattern_block

    # Determine target blocks based on position type
    if r == 1 and c == 1:  # Center
        # Anti-diagonal + middle-right
        target_blocks = [(0, 2), (1, 1), (1, 2), (2, 0)]
    elif (r == 1) != (c == 1):  # Edge (exactly one coordinate is 1)
        # All 4 edges + bottom-left corner
        target_blocks = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0)]
    else:  # Corner
        # Source + neighbor in same row + (1,2) + (2,0)
        # Neighbor is the edge block in same row
        target_blocks = [(r, c), (r, 1), (1, 2), (2, 0)]

    # Copy pattern to target blocks
    for target_r, target_c in target_blocks:
        for (rel_r, rel_c), value in pattern.items():
            abs_r = target_r * 3 + rel_r
            abs_c = target_c * 3 + rel_c
            if 0 <= abs_r < 9 and 0 <= abs_c < 9:
                result[abs_r][abs_c] = value

    return result
