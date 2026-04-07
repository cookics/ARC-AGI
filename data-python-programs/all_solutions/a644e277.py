def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid divided by separator lines into rectangular blocks
    2. Output extracts specific middle blocks and arranges them with borders/separators
    3. Each output data block is a direct copy of an input block
    4. Output has: border rows/cols (corners=background, edges=separator), separators, data blocks

    Procedure:
    1. Identify separator value and find separator rows/columns
    2. Extract sections between separators
    3. Select middle blocks (typically indices 1,2 for both rows and cols)
    4. Build output with proper borders and separators
    """
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Find separator value - it should form lines
    # Check each unique value to see which forms the most complete lines
    all_values = set(grid[r][c] for r in range(rows) for c in range(cols))

    separator = None
    max_line_count = 0
    for val in all_values:
        # Count how many rows are mostly this value
        row_count = sum(1 for r in range(rows)
                       if sum(1 for c in range(cols) if grid[r][c] == val) >= cols * 0.8)
        # Count how many columns are mostly this value
        col_count = sum(1 for c in range(cols)
                       if sum(1 for r in range(rows) if grid[r][c] == val) >= rows * 0.8)
        total = row_count + col_count
        if total > max_line_count:
            max_line_count = total
            separator = val

    # Find separator rows
    sep_rows = []
    for r in range(rows):
        if sum(1 for c in range(cols) if grid[r][c] == separator) >= cols * 0.8:
            sep_rows.append(r)

    # Find separator columns
    sep_cols = []
    for c in range(cols):
        if sum(1 for r in range(rows) if grid[r][c] == separator) >= rows * 0.8:
            sep_cols.append(c)

    # Build row sections
    row_sections = []
    start = 0
    for sep in sep_rows:
        if sep > start:
            row_sections.append(list(range(start, sep)))
        start = sep + 1
    if start < rows:
        row_sections.append(list(range(start, rows)))

    # Build column sections
    col_sections = []
    start = 0
    for sep in sep_cols:
        if sep > start:
            col_sections.append(list(range(start, sep)))
        start = sep + 1
    if start < cols:
        col_sections.append(list(range(start, cols)))

    # Find background value (most common non-separator)
    background = None
    value_counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    for val, _ in value_counts.most_common():
        if val != separator:
            background = val
            break

    # Determine which blocks to extract
    nr = len(row_sections)
    nc = len(col_sections)

    # Select middle sections
    if nr >= 5:
        row_indices = [1, 2]
    elif nr == 4:
        row_indices = [1]
    elif nr >= 2:
        row_indices = [1]
    else:
        row_indices = [0]

    if nc >= 5:
        col_indices = [1, 2]
    elif nc >= 3:
        col_indices = [1, 2]
    else:
        col_indices = [0, 1] if nc >= 2 else [0]

    # Extract the blocks
    extracted_blocks = []
    for ri in row_indices:
        row_of_blocks = []
        for ci in col_indices:
            rows_in_section = row_sections[ri]
            cols_in_section = col_sections[ci]
            block = [[grid[r][c] for c in cols_in_section] for r in rows_in_section]
            row_of_blocks.append(block)
        extracted_blocks.append(row_of_blocks)

    # Get block dimensions
    block_h = len(extracted_blocks[0][0])
    block_w = len(extracted_blocks[0][0][0])
    n_block_rows = len(extracted_blocks)
    n_block_cols = len(extracted_blocks[0])

    # Build output
    out_h = 1 + n_block_rows * block_h + (n_block_rows - 1) + 1
    out_w = 1 + n_block_cols * block_w + (n_block_cols - 1) + 1

    result = [[separator for _ in range(out_w)] for _ in range(out_h)]

    # Set corners to background
    result[0][0] = background
    result[0][out_w-1] = background
    result[out_h-1][0] = background
    result[out_h-1][out_w-1] = background

    # Place blocks
    for bi in range(n_block_rows):
        for bj in range(n_block_cols):
            block = extracted_blocks[bi][bj]
            start_r = 1 + bi * (block_h + 1)
            start_c = 1 + bj * (block_w + 1)
            for i in range(block_h):
                for j in range(block_w):
                    result[start_r + i][start_c + j] = block[i][j]

    return result
