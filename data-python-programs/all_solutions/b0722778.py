def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid is divided by rows and columns of all zeros into a grid of 2×2 blocks
    2. Each row band has 3 blocks: left, middle, right
    3. Output has 2 columns, derived from one of the three blocks after transformation
    4. Block selection: if left H-flipped = middle use right; else if left/right same values use middle; else if left/middle same values use right
    5. Transformation on right: if middle has identical rows rotate clockwise; else if middle first row uniform rotate counter-clockwise; else rotate clockwise
    6. Transformation on middle: if left V-flipped = right reverse rows; else duplicate first row

    Procedure:
    1. Process rows in pairs (skip separator rows of all zeros)
    2. Extract left/middle/right 2×2 blocks from each pair
    3. Apply block selection and transformation rules
    4. Build output with transformed blocks
    """

    def horizontal_flip(block):
        """Flip block horizontally (reverse each row)"""
        return [row[::-1] for row in block]

    def vertical_flip(block):
        """Flip block vertically (reverse row order)"""
        return block[::-1]

    def rotate_clockwise(block):
        """Rotate 2×2 block 90° clockwise: [[a,b],[c,d]] -> [[c,a],[d,b]]"""
        return [[block[1][0], block[0][0]], [block[1][1], block[0][1]]]

    def rotate_counterclockwise(block):
        """Rotate 2×2 block 90° counter-clockwise: [[a,b],[c,d]] -> [[b,d],[a,c]]"""
        return [[block[0][1], block[1][1]], [block[0][0], block[1][0]]]

    def get_value_set(block):
        """Get set of all values in block"""
        return set(val for row in block for val in row)

    def blocks_equal(block1, block2):
        """Check if two blocks are identical"""
        return block1 == block2

    result = []

    # Process each row
    i = 0
    while i < len(grid):
        row = grid[i]

        # Check if this is a separator row (all zeros)
        if all(x == 0 for x in row):
            result.append([0, 0])
            i += 1
            continue

        # Check if next row is also non-zero (should be part of a 2×2 block)
        if i + 1 < len(grid) and not all(x == 0 for x in grid[i + 1]):
            # Extract 2×2 blocks
            left_block = [[grid[i][0], grid[i][1]], [grid[i+1][0], grid[i+1][1]]]
            middle_block = [[grid[i][3], grid[i][4]], [grid[i+1][3], grid[i+1][4]]]
            right_block = [[grid[i][7], grid[i][8]], [grid[i+1][7], grid[i+1][8]]]

            # Determine which block to use
            left_h_flip = horizontal_flip(left_block)

            if blocks_equal(left_h_flip, middle_block):
                # Use right block
                if middle_block[0] == middle_block[1]:  # Identical rows
                    output_block = rotate_clockwise(right_block)
                else:  # Non-identical rows
                    output_block = rotate_counterclockwise(right_block)
            elif get_value_set(left_block) == get_value_set(right_block):
                # Use middle block
                left_v_flip = vertical_flip(left_block)
                if blocks_equal(left_v_flip, right_block):
                    output_block = vertical_flip(middle_block)
                else:
                    # Duplicate first row
                    output_block = [middle_block[0][:], middle_block[0][:]]
            elif get_value_set(left_block) == get_value_set(middle_block):
                # Use right block
                if middle_block[0] == middle_block[1]:  # Identical rows
                    output_block = rotate_clockwise(right_block)
                else:  # Non-identical rows
                    output_block = rotate_counterclockwise(right_block)
            else:
                # Default: use right block with clockwise rotation
                output_block = rotate_clockwise(right_block)

            # Add both rows of output block
            result.append(output_block[0])
            result.append(output_block[1])
            i += 2
        else:
            # Single row (shouldn't normally happen in this problem)
            result.append([grid[i][7], grid[i][8]])
            i += 1

    return result
