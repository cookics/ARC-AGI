def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has background, 2x2 blocks (two types), and connector lines
    2. Find longest vertical line of connectors
    3. Transform 2x2 blocks at ONE specific end of the grid (top OR bottom)
    4. Also transform the long connector line and nearby connectors

    Procedure:
    1. Identify background, connector, and block values
    2. Find longest vertical line of connectors
    3. Determine which end (top/bottom) has blocks to transform
    4. Transform those blocks to 3 and connectors to 5
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find background
    all_colors = [grid[r][c] for r in range(rows) for c in range(cols)]
    color_counts = Counter(all_colors)
    background = color_counts.most_common(1)[0][0]

    # Find all 2x2 blocks
    def find_2x2_blocks(val):
        blocks = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                if all(grid[r+dr][c+dc] == val for dr in range(2) for dc in range(2)):
                    blocks.append((r, c))
        return blocks

    # Identify connector and block values
    non_bg = [c for c in color_counts if c != background]
    if len(non_bg) < 2:
        return result

    block_counts = {val: len(find_2x2_blocks(val)) for val in non_bg}
    connector_val = min(non_bg, key=lambda v: block_counts[v])
    block_vals = [v for v in non_bg if v != connector_val]

    # Find all connector cells
    connector_cells = {(r, c) for r in range(rows) for c in range(cols)
                       if grid[r][c] == connector_val}

    # Find longest lines in all directions
    all_lines = []

    # Vertical lines
    for c in range(cols):
        line = []
        for r in range(rows):
            if (r, c) in connector_cells:
                line.append((r, c))
            elif line:
                if len(line) >= 3:
                    all_lines.append(line)
                line = []
        if len(line) >= 3:
            all_lines.append(line)

    # Horizontal lines
    for r in range(rows):
        line = []
        for c in range(cols):
            if (r, c) in connector_cells:
                line.append((r, c))
            elif line:
                if len(line) >= 3:
                    all_lines.append(line)
                line = []
        if len(line) >= 3:
            all_lines.append(line)

    if not all_lines:
        return result

    # Sort by length and pick the longest line
    all_lines.sort(key=len, reverse=True)
    longest_line = all_lines[0]

    # Determine which end of the grid to transform based on where the line is
    line_center_row = sum(r for r, c in longest_line) / len(longest_line)
    mid_row = rows // 2

    # If line is in top half, transform top blocks; if in bottom half, transform bottom
    transform_bottom = line_center_row >= mid_row

    # Find all blocks of each type
    blocks_by_type = {val: find_2x2_blocks(val) for val in block_vals}

    # KEY INSIGHT: Transform the block type with MORE blocks
    if len(block_vals) >= 2:
        block_val_to_transform = max(block_vals, key=lambda v: len(blocks_by_type[v]))
    elif len(block_vals) == 1:
        block_val_to_transform = block_vals[0]
    else:
        return result

    blocks = blocks_by_type[block_val_to_transform]

    # Find blocks in the target region (top or bottom half)
    target_blocks = []
    if transform_bottom:
        # Transform blocks in bottom half
        target_blocks = [b for b in blocks if b[0] >= mid_row]
    else:
        # Transform blocks in top half
        target_blocks = [b for b in blocks if b[0] < mid_row]

    blocks_to_transform = set()
    connectors_to_transform = set(longest_v_line)

    if target_blocks:
        # Transform these blocks
        for br, bc in target_blocks:
            blocks_to_transform.add((br, bc, block_val_to_transform))

        # Find connector cells IMMEDIATELY adjacent to these blocks
        for br, bc in target_blocks:
            # Check cells in and around the 2x2 block
            for dr in range(-1, 3):
                for dc in range(-1, 3):
                    r, c = br + dr, bc + dc
                    if 0 <= r < rows and 0 <= c < cols and (r, c) in connector_cells:
                        # Only add if it's touching the block
                        touches_block = False
                        for bdr in range(2):
                            for bdc in range(2):
                                block_r, block_c = br + bdr, bc + bdc
                                if abs(r - block_r) <= 1 and abs(c - block_c) <= 1:
                                    touches_block = True
                        if touches_block:
                            connectors_to_transform.add((r, c))

    # Transform blocks to 3
    for br, bc, _ in blocks_to_transform:
        for dr in range(2):
            for dc in range(2):
                result[br+dr][bc+dc] = 3

    # Transform connectors to 5
    for r, c in connectors_to_transform:
        result[r][c] = 5

    return result
