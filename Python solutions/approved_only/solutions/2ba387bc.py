def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large grid containing scattered 4x4 rectangular blocks of different colors (non-zero values).
    2. Each 4x4 block is either "solid" (completely filled with the same color) or "hollow" (border filled with color, center cells are 0).
    3. Output is a compact arrangement where hollow blocks are paired with solid blocks side by side.
    4. Hollow blocks appear on the left, solid blocks on the right in each pair.
    5. Blocks are arranged in spatial order based on their original position (top to bottom, left to right).
    6. If there are unequal numbers of hollow and solid blocks, zeros fill the empty positions.

    Procedure:
    1. Scan the input grid to find all 4x4 blocks and extract their positions and contents.
    2. Classify each block as either hollow (center cells are 0) or solid (all cells filled).
    3. Sort all blocks by their spatial position (row first, then column).
    4. Separate blocks into hollow and solid categories while maintaining order.
    5. Create pairs by matching hollow blocks with solid blocks in sequence.
    6. Generate the output by concatenating each pair horizontally (4 rows per pair).
    """

    def find_blocks(grid):
        """Find all 4x4 blocks and their positions"""
        blocks = []
        rows, cols = len(grid), len(grid[0])

        for r in range(rows - 3):
            for c in range(cols - 3):
                # Check if this is the top-left of a 4x4 block
                value = grid[r][c]
                if value != 0:
                    # Check if all border positions have the same value
                    is_block = True
                    for i in range(4):
                        for j in range(4):
                            if i == 0 or i == 3 or j == 0 or j == 3:  # Border positions
                                if grid[r + i][c + j] != value:
                                    is_block = False
                                    break
                        if not is_block:
                            break

                    if is_block:
                        # Extract the 4x4 block
                        block = []
                        for i in range(4):
                            row = []
                            for j in range(4):
                                row.append(grid[r + i][c + j])
                            block.append(row)
                        blocks.append((r, c, value, block))

        return blocks

    def is_hollow(block):
        """Check if a 4x4 block is hollow (has 0s in center)"""
        return (
            block[1][1] == 0
            and block[1][2] == 0
            and block[2][1] == 0
            and block[2][2] == 0
        )

    def create_solid_block(value):
        """Create a 4x4 solid block with given value"""
        return [[value] * 4 for _ in range(4)]

    def create_zero_block():
        """Create a 4x4 block of zeros"""
        return [[0] * 4 for _ in range(4)]

    # Find all blocks
    blocks = find_blocks(grid)

    # Sort blocks by position (row first, then column)
    blocks.sort(key=lambda x: (x[0], x[1]))

    # Separate hollow and solid blocks
    hollow_blocks = []
    solid_blocks = []

    for r, c, value, block in blocks:
        if is_hollow(block):
            hollow_blocks.append(block)
        else:
            solid_blocks.append(create_solid_block(value))

    # Create pairs
    pairs = []

    # Pair hollow blocks with solid blocks
    for i in range(len(hollow_blocks)):
        if i < len(solid_blocks):
            pairs.append((hollow_blocks[i], solid_blocks[i]))
        else:
            pairs.append((hollow_blocks[i], create_zero_block()))

    # Add remaining solid blocks paired with zeros
    for i in range(len(hollow_blocks), len(solid_blocks)):
        pairs.append((create_zero_block(), solid_blocks[i]))

    # Create output
    result = []
    for left_block, right_block in pairs:
        for i in range(4):
            row = left_block[i] + right_block[i]
            result.append(row)

    return result
