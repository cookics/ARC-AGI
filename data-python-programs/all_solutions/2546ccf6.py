def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided by separator lines (horizontal and vertical lines filled with same non-zero value)
    2. Output is the same grid where empty rectangular sections are filled with patterns
    3. The grid is partitioned into rectangular blocks by these separator lines
    4. Some blocks contain non-zero patterns (besides separators), while others are empty (all zeros)
    5. Empty blocks get filled with horizontally flipped versions of adjacent non-empty blocks
    6. Within each horizontal band (row of blocks), typically one block has a pattern

    Procedure:
    1. Identify all horizontal separator lines (rows where all cells have same non-zero value)
    2. Identify all vertical separator lines (columns where all cells have same non-zero value)
    3. Use separator positions to partition grid into rectangular blocks
    4. For each horizontal band of blocks, find non-empty blocks (containing pattern)
    5. For bands with exactly one non-empty block, determine which adjacent empty block to fill
    6. Fill the adjacent empty block with the horizontally flipped version of the pattern
    """

    def flip_horizontal(block):
        """Flip a block horizontally"""
        return [row[::-1] for row in block]

    def fill_block(grid, bounds, content):
        """Fill a block in the grid with given content"""
        r_start, r_end, c_start, c_end = bounds
        for i, r in enumerate(range(r_start, r_end)):
            for j, c in enumerate(range(c_start, c_end)):
                grid[r][c] = content[i][j]

    result = [row[:] for row in grid]  # Deep copy
    rows, cols = len(grid), len(grid[0])

    # Find separator lines - they are lines where all values are the same and non-zero
    h_separators = []
    v_separators = []

    # Find horizontal separators
    for r in range(rows):
        if all(grid[r][c] == grid[r][0] and grid[r][0] != 0 for c in range(cols)):
            h_separators.append(r)

    # Find vertical separators
    for c in range(cols):
        if all(grid[r][c] == grid[0][c] and grid[0][c] != 0 for r in range(rows)):
            v_separators.append(c)

    # Get block boundaries (including grid edges)
    h_boundaries = [-1] + h_separators + [rows]
    v_boundaries = [-1] + v_separators + [cols]

    # Create blocks
    blocks = []
    for i in range(len(h_boundaries) - 1):
        block_row = []
        for j in range(len(v_boundaries) - 1):
            r_start = h_boundaries[i] + 1
            r_end = h_boundaries[i + 1]
            c_start = v_boundaries[j] + 1
            c_end = v_boundaries[j + 1]

            # Extract block content
            block = []
            for r in range(r_start, r_end):
                block.append(grid[r][c_start:c_end])

            block_row.append(
                {
                    "content": block,
                    "bounds": (r_start, r_end, c_start, c_end),
                    "is_empty": all(
                        grid[r][c] == 0
                        for r in range(r_start, r_end)
                        for c in range(c_start, c_end)
                    ),
                }
            )
        blocks.append(block_row)

    # Fill empty blocks with flipped adjacent blocks
    # Only fill ONE empty block per non-empty block that is the ONLY non-empty block in its row
    for i in range(len(blocks)):
        # Count non-empty blocks in this row
        non_empty_blocks_in_row = [
            j for j in range(len(blocks[i])) if not blocks[i][j]["is_empty"]
        ]

        # Only process rows with exactly one non-empty block
        if len(non_empty_blocks_in_row) == 1:
            non_empty_pos = non_empty_blocks_in_row[0]

            # Determine direction based on position of non-empty block
            num_blocks = len(blocks[i])
            is_left_half = non_empty_pos < num_blocks // 2

            # If in left half, prefer filling to the right; if in right half, prefer filling to the left
            if is_left_half:
                # Fill right first, then left
                if (
                    non_empty_pos < len(blocks[i]) - 1
                    and blocks[i][non_empty_pos + 1]["is_empty"]
                ):
                    flipped_content = flip_horizontal(
                        blocks[i][non_empty_pos]["content"]
                    )
                    fill_block(
                        result, blocks[i][non_empty_pos + 1]["bounds"], flipped_content
                    )
                elif non_empty_pos > 0 and blocks[i][non_empty_pos - 1]["is_empty"]:
                    flipped_content = flip_horizontal(
                        blocks[i][non_empty_pos]["content"]
                    )
                    fill_block(
                        result, blocks[i][non_empty_pos - 1]["bounds"], flipped_content
                    )
            else:
                # Fill left first, then right
                if non_empty_pos > 0 and blocks[i][non_empty_pos - 1]["is_empty"]:
                    flipped_content = flip_horizontal(
                        blocks[i][non_empty_pos]["content"]
                    )
                    fill_block(
                        result, blocks[i][non_empty_pos - 1]["bounds"], flipped_content
                    )
                elif (
                    non_empty_pos < len(blocks[i]) - 1
                    and blocks[i][non_empty_pos + 1]["is_empty"]
                ):
                    flipped_content = flip_horizontal(
                        blocks[i][non_empty_pos]["content"]
                    )
                    fill_block(
                        result, blocks[i][non_empty_pos + 1]["bounds"], flipped_content
                    )

    return result
