def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided by separator lines (all 6s) into rectangular blocks
    2. Output rearranges these blocks based on the layout type
    3. For 2×2 layouts: blocks stack vertically in order top-left, bottom-right, top-right, bottom-left
    4. For vertical layouts (N rows, 1 column): blocks arrange horizontally in reverse order
    5. For horizontal layouts (1 row, N columns): blocks stack vertically in same order
    6. Separators between blocks remain as rows or columns of 6s in the output

    Procedure:
    1. Identify horizontal and vertical separator lines (rows/columns that are all 6s)
    2. Extract all rectangular blocks bounded by separators
    3. Determine the layout type based on number of block rows and columns
    4. Rearrange blocks according to the transformation rule for that layout type
    5. Insert separator lines between blocks in the output
    """

    rows, cols = len(grid), len(grid[0])

    # Find horizontal separators (rows that are all 6's)
    h_separators = []
    for r in range(rows):
        if all(grid[r][c] == 6 for c in range(cols)):
            h_separators.append(r)

    # Find vertical separators (columns that are all 6's)
    v_separators = []
    for c in range(cols):
        if all(grid[r][c] == 6 for r in range(rows)):
            v_separators.append(c)

    # Extract blocks
    h_ranges = []
    prev = 0
    for sep in h_separators:
        if prev < sep:
            h_ranges.append((prev, sep))
        prev = sep + 1
    if prev < rows:
        h_ranges.append((prev, rows))

    v_ranges = []
    prev = 0
    for sep in v_separators:
        if prev < sep:
            v_ranges.append((prev, sep))
        prev = sep + 1
    if prev < cols:
        v_ranges.append((prev, cols))

    # Extract all blocks
    blocks = []
    for hr_start, hr_end in h_ranges:
        row_blocks = []
        for vr_start, vr_end in v_ranges:
            block = []
            for r in range(hr_start, hr_end):
                block.append(grid[r][vr_start:vr_end])
            row_blocks.append(block)
        blocks.append(row_blocks)

    # Determine layout and rearrange
    num_block_rows = len(blocks)
    num_block_cols = len(blocks[0]) if blocks else 0

    result = []

    if num_block_rows == 2 and num_block_cols == 2:
        # 2x2 layout: order is top-left, bottom-right, top-right, bottom-left
        order = [(0, 0), (1, 1), (0, 1), (1, 0)]
        for i, (br, bc) in enumerate(order):
            if i > 0:
                # Add separator row
                block_width = len(blocks[0][0][0])
                result.append([6] * block_width)
            result.extend(blocks[br][bc])

    elif num_block_rows > 1 and num_block_cols == 1:
        # Vertical layout -> horizontal layout (reverse order)
        block_height = len(blocks[0][0])
        for r in range(block_height):
            row = []
            for i in reversed(range(num_block_rows)):
                if len(row) > 0:
                    row.append(6)
                row.extend(blocks[i][0][r])
            result.append(row)

    elif num_block_rows == 1 and num_block_cols > 1:
        # Horizontal layout -> vertical layout (same order)
        for c in range(num_block_cols):
            if c > 0:
                # Add separator row
                block_width = len(blocks[0][0][0])
                result.append([6] * block_width)
            result.extend(blocks[0][c])

    return result
