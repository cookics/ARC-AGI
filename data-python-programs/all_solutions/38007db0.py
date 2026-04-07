def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid is divided by separator lines (rows/columns where all values are the same)
    2. This creates a grid of blocks
    3. For each row of blocks, one block is unique (different from the others)
    4. Output consists of the unique block from each row, preserving the separator structure

    Procedure:
    1. Identify separator rows and columns
    2. Extract blocks between separators
    3. For each row of blocks, find the unique block
    4. Assemble output with unique blocks and separators
    """

    n_rows = len(grid)
    n_cols = len(grid[0])

    # Identify separator rows (all values are the same)
    sep_rows = []
    for i in range(n_rows):
        if len(set(grid[i])) == 1:
            sep_rows.append(i)

    # Identify separator columns (all values are the same)
    sep_cols = []
    for j in range(n_cols):
        col_values = [grid[i][j] for i in range(n_rows)]
        if len(set(col_values)) == 1:
            sep_cols.append(j)

    # Divide grid into row blocks (between consecutive separators)
    row_ranges = []
    for i in range(len(sep_rows) - 1):
        row_ranges.append((sep_rows[i] + 1, sep_rows[i + 1]))

    # Divide grid into column blocks (between consecutive separators)
    col_ranges = []
    for j in range(len(sep_cols) - 1):
        col_ranges.append((sep_cols[j] + 1, sep_cols[j + 1]))

    # Extract all blocks
    blocks = []
    for row_start, row_end in row_ranges:
        row_of_blocks = []
        for col_start, col_end in col_ranges:
            block = []
            for i in range(row_start, row_end):
                block.append(grid[i][col_start:col_end])
            row_of_blocks.append(block)
        blocks.append(row_of_blocks)

    # For each row of blocks, find the unique block
    unique_blocks = []
    for row_of_blocks in blocks:
        # Convert blocks to tuples for comparison
        block_tuples = [tuple(tuple(row) for row in block) for block in row_of_blocks]

        # Find the unique block (appears exactly once)
        unique_block = None
        for i, block_tuple in enumerate(block_tuples):
            if block_tuples.count(block_tuple) == 1:
                unique_block = row_of_blocks[i]
                break

        unique_blocks.append(unique_block)

    # Assemble output
    result = []
    sep_value = grid[sep_rows[0]][0]

    # Determine output width
    block_width = len(unique_blocks[0][0])
    output_width = block_width + 2  # block width + 2 boundary separators

    for unique_block in unique_blocks:
        # Add separator row before this block
        result.append([sep_value] * output_width)

        # Add block rows with boundary separators
        for row in unique_block:
            result.append([sep_value] + row + [sep_value])

    # Add final separator row
    result.append([sep_value] * output_width)

    return result
