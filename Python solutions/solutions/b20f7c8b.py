def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid split into block section (4 blocks in 2×2 arrangement) and marker section (scattered colors on 8-background)
    2. Each 5×5 block either has pattern (contains 1) or is uniform (single color)
    3. Transformation flips pattern state AND reassigns colors based on markers
    4. Pattern copying follows diagonal rule when blocks gain patterns

    Procedure:
    1. Detect sections by finding 8-background region
    2. Find exactly 4 blocks (5×5 each) in block section
    3. Extract markers and sort by average row position
    4. For each block: flip pattern state (uniform↔pattern) and reassign color
    5. When block becomes patterned, copy 1-positions from diagonal opposite or any patterned block
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find section boundary by detecting columns with many 8s
    col_8_counts = [sum(1 for r in range(rows) if grid[r][c] == 8) for c in range(cols)]

    # Determine marker section (has many 8s)
    threshold = rows // 2
    if sum(1 for c in range(cols // 2) if col_8_counts[c] > threshold) > sum(1 for c in range(cols // 2, cols) if col_8_counts[c] > threshold):
        # Left has 8s (markers), right has blocks
        marker_cols = [c for c in range(cols) if col_8_counts[c] > threshold]
        block_start = max(marker_cols) + 1 if marker_cols else 0
        block_end = cols
        marker_start = 0
        marker_end = min(marker_cols[-1] + 1 if marker_cols else cols, cols)
    else:
        # Right has 8s (markers), left has blocks
        marker_cols = [c for c in range(cols) if col_8_counts[c] > threshold]
        marker_start = min(marker_cols) if marker_cols else cols
        marker_end = cols
        block_start = 0
        block_end = marker_start

    # Extract markers sorted by vertical position
    marker_positions = {}
    for r in range(rows):
        for c in range(marker_start, marker_end):
            if c < cols and grid[r][c] not in [0, 8]:
                color = grid[r][c]
                if color not in marker_positions:
                    marker_positions[color] = []
                marker_positions[color].append(r)

    markers = sorted(marker_positions.keys(), key=lambda c: sum(marker_positions[c]) / len(marker_positions[c]))

    # Find block starting rows by looking for rows with block content
    block_rows = []
    for r in range(rows - 4):
        # Check if this row starts a block (has 5+ consecutive filled cells in block region)
        row_content = sum(1 for c in range(block_start, min(block_end, cols)) if grid[r][c] not in [0, 8])
        if row_content >= 5:
            # Check if we can form a 5×5 block starting here
            block_cells = sum(1 for rr in range(r, min(r + 5, rows)) for cc in range(block_start, min(block_end, cols)) if grid[rr][cc] not in [0, 8])
            if block_cells >= 40:  # At least 40 cells across 2 blocks
                if not block_rows or r - block_rows[-1] >= 6:
                    block_rows.append(r)
                    if len(block_rows) == 2:
                        break

    # Find block starting columns
    block_cols = []
    for c in range(block_start, block_end - 4):
        col_content = sum(1 for r in range(rows) if grid[r][c] not in [0, 8])
        if col_content >= 5:
            if not block_cols or c - block_cols[-1] >= 6:
                block_cols.append(c)
                if len(block_cols) == 2:
                    break

    # Extract blocks at detected positions
    def get_block_at(r0, c0):
        if r0 + 5 > rows or c0 + 5 > cols:
            return None
        data = [[grid[r][c] for c in range(c0, c0 + 5)] for r in range(r0, r0 + 5)]
        vals = [v for row in data for v in row if v not in [0, 8]]
        if len(vals) < 20:
            return None
        has_pattern = 1 in vals
        if has_pattern:
            color = max((v for v in vals if v != 1), key=vals.count, default=2)
        else:
            color = max(set(vals), key=vals.count)
        return {'r0': r0, 'c0': c0, 'data': data, 'color': color, 'pattern': has_pattern}

    blocks = []
    if len(block_rows) >= 2 and len(block_cols) >= 2:
        for r in block_rows:
            for c in block_cols:
                block = get_block_at(r, c)
                if block:
                    blocks.append(block)

    if len(blocks) != 4 or len(markers) != 4:
        return result

    # Sort blocks by position (top-left, top-right, bottom-left, bottom-right)
    blocks.sort(key=lambda b: (b['r0'], b['c0']))

    # Determine new colors for each block
    input_colors = [b['color'] for b in blocks]

    # First pass: determine which blocks become 2 (collision with non-2 markers)
    temp_colors = []
    blocks_needing_markers = []
    for i, block in enumerate(blocks):
        if block['color'] in markers and block['color'] != 2:
            temp_colors.append(2)
        else:
            temp_colors.append(None)  # Needs assignment
            blocks_needing_markers.append(i)

    # Assign markers to blocks that need them
    # Pattern depends on how many blocks need assignment
    if len(blocks_needing_markers) == 4:
        marker_indices = [0, 2, 1, 3]
    elif len(blocks_needing_markers) == 3:
        marker_indices = [0, 3, 2]
    elif len(blocks_needing_markers) == 2:
        marker_indices = [1, 2]
    elif len(blocks_needing_markers) == 1:
        marker_indices = [0]
    else:
        marker_indices = []

    # Assign markers
    for idx, block_i in enumerate(blocks_needing_markers):
        if idx < len(marker_indices):
            temp_colors[block_i] = markers[marker_indices[idx]]

    new_colors = temp_colors

    # Get pattern states
    old_patterns = [b['pattern'] for b in blocks]
    new_patterns = [not p for p in old_patterns]  # Flip pattern state

    # Write transformed blocks
    for i, block in enumerate(blocks):
        r0, c0 = block['r0'], block['c0']
        new_color = new_colors[i]
        needs_pattern = new_patterns[i]

        if needs_pattern:
            # Copy 1s from diagonal opposite or any block with pattern
            diag_idx = 3 - i  # 0↔3, 1↔2
            source = blocks[diag_idx] if old_patterns[diag_idx] else next((b for j, b in enumerate(blocks) if old_patterns[j]), None)

            if source:
                for dr in range(5):
                    for dc in range(5):
                        src_val = source['data'][dr][dc]
                        result[r0 + dr][c0 + dc] = 1 if src_val == 1 else new_color
            else:
                # No source pattern, fill solid
                for dr in range(5):
                    for dc in range(5):
                        result[r0 + dr][c0 + dc] = new_color
        else:
            # Fill solid
            for dr in range(5):
                for dc in range(5):
                    result[r0 + dr][c0 + dc] = new_color

    return result
