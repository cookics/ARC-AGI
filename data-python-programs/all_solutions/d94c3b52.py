def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid is 17×25, divided into 4 row-blocks × 6 col-blocks of 3×3 cells separated by 0s
    2. Some 3×3 blocks contain 8s as markers indicating a reference pattern
    3. For each row of blocks, find pairs of blocks with matching patterns
    4. Transform blocks between pairs: matching marker pattern → 8, others → 7

    Procedure:
    1. Extract all 3×3 blocks from grid
    2. Find marker block(s) containing 8s and extract their pattern
    3. For each row of blocks, find blocks matching the marker pattern
    4. If 2+ matches exist in a row, transform blocks between leftmost and rightmost match
    """

    result = [row[:] for row in grid]

    # Define block boundaries
    def get_block(row_idx, col_idx):
        """Get 3×3 block at given row and column indices"""
        r_start = 1 + row_idx * 4
        c_start = 1 + col_idx * 4
        return [[grid[r_start + i][c_start + j] for j in range(3)] for i in range(3)]

    def set_block(row_idx, col_idx, block):
        """Set 3×3 block at given row and column indices"""
        r_start = 1 + row_idx * 4
        c_start = 1 + col_idx * 4
        for i in range(3):
            for j in range(3):
                result[r_start + i][c_start + j] = block[i][j]

    def get_pattern(block):
        """Extract binary pattern (1 for non-zero, 0 for zero)"""
        return tuple(tuple(1 if cell != 0 else 0 for cell in row) for row in block)

    def has_marker(block):
        """Check if block contains 8s"""
        return any(8 in row for row in block)

    # Find marker block and extract its pattern
    marker_pattern = None
    marker_row_idx = None
    marker_col_idx = None

    for r_idx in range(4):
        for c_idx in range(6):
            block = get_block(r_idx, c_idx)
            if has_marker(block):
                marker_pattern = get_pattern(block)
                marker_row_idx = r_idx
                marker_col_idx = c_idx
                break
        if marker_pattern:
            break

    if not marker_pattern:
        return result

    # First pass: Process marker row to determine transformed columns
    marker_row_transformed = []

    if marker_row_idx is not None:
        # Find matching blocks in marker row
        matching_indices = []
        for c_idx in range(6):
            if c_idx == marker_col_idx:
                continue
            block = get_block(marker_row_idx, c_idx)
            if get_pattern(block) == marker_pattern:
                matching_indices.append(c_idx)

        # Transform between leftmost match and marker
        if matching_indices:
            # Include marker in the range calculation
            all_indices = matching_indices + [marker_col_idx]
            left = min(all_indices)
            right = max(all_indices)

            for c_idx in range(left, right + 1):
                if c_idx == marker_col_idx:
                    continue  # Don't transform or track the marker itself

                block = get_block(marker_row_idx, c_idx)
                pattern = get_pattern(block)

                if pattern == marker_pattern:
                    new_block = [[8 if cell != 0 else 0 for cell in row] for row in block]
                else:
                    new_block = [[7 if cell == 1 else cell for cell in row] for row in block]
                set_block(marker_row_idx, c_idx, new_block)
                marker_row_transformed.append(c_idx)

    # Get marker row's matching block columns (for rows with no matches)
    marker_row_match_cols = []
    for c_idx in range(6):
        if c_idx == marker_col_idx:
            continue
        block = get_block(marker_row_idx, c_idx)
        if get_pattern(block) == marker_pattern:
            marker_row_match_cols.append(c_idx)

    # Get marker row's pair boundaries
    marker_pair_left = None
    marker_pair_right = None
    if marker_row_match_cols:
        all_pair_indices = marker_row_match_cols + [marker_col_idx]
        marker_pair_left = min(all_pair_indices)
        marker_pair_right = max(all_pair_indices)

    # Second pass: Process other rows
    for r_idx in range(4):
        if r_idx == marker_row_idx:
            continue

        # Find matching blocks in this row
        matching_indices = []
        for c_idx in range(6):
            block = get_block(r_idx, c_idx)
            if get_pattern(block) == marker_pattern:
                matching_indices.append(c_idx)

        if len(matching_indices) >= 2:
            # Transform between leftmost and rightmost match
            left = min(matching_indices)
            right = max(matching_indices)

            for c_idx in range(left, right + 1):
                block = get_block(r_idx, c_idx)
                pattern = get_pattern(block)

                if pattern == marker_pattern:
                    new_block = [[8 if cell != 0 else 0 for cell in row] for row in block]
                else:
                    new_block = [[7 if cell == 1 else cell for cell in row] for row in block]
                set_block(r_idx, c_idx, new_block)
        elif len(matching_indices) == 1:
            # One matching block: check if it's within marker pair range
            match_idx = matching_indices[0]

            if marker_pair_left is not None and marker_pair_right is not None:
                if marker_pair_left <= match_idx <= marker_pair_right:
                    # Transform from match to the edge of the pair
                    left = match_idx
                    right = marker_pair_right

                    for c_idx in range(left, right + 1):
                        block = get_block(r_idx, c_idx)
                        pattern = get_pattern(block)

                        if pattern == marker_pattern:
                            new_block = [[8 if cell != 0 else 0 for cell in row] for row in block]
                        else:
                            new_block = [[7 if cell == 1 else cell for cell in row] for row in block]
                        set_block(r_idx, c_idx, new_block)
        else:
            # No matches: only transform if marker is at row 0, or if current row is above marker
            should_transform = (marker_row_idx == 0) or (r_idx < marker_row_idx)

            if should_transform:
                # Transform at marker row's matching block columns AND marker column
                transform_cols = marker_row_match_cols + [marker_col_idx]

                for c_idx in transform_cols:
                    block = get_block(r_idx, c_idx)
                    pattern = get_pattern(block)

                    # Count non-zero cells
                    non_zero_count = sum(1 for row in block for cell in row if cell != 0)

                    # Only transform if block has significant content (at least 3 non-zero cells)
                    if non_zero_count >= 3:
                        if pattern == marker_pattern:
                            new_block = [[8 if cell != 0 else 0 for cell in row] for row in block]
                        else:
                            new_block = [[7 if cell == 1 else cell for cell in row] for row in block]
                        set_block(r_idx, c_idx, new_block)

    return result
