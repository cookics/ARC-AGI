def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a sparse marker pattern (top-left area) with values at specific rows/columns
    2. Input has filled rectangular blocks elsewhere
    3. Output replicates blocks based on marker patterns
    4. Each marker row specifies values for blocks at calculated positions

    Procedure:
    1. Find background color (most common value)
    2. Extract sparse markers from rows 1,3,5,... at columns 1,3,5,...
    3. Find existing filled rectangular blocks
    4. Match marker row to existing blocks by comparing value sets
    5. Calculate block positions for each marker row
    6. Fill blocks with marker values
    """
    from collections import Counter
    import copy

    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find background color
    flat = [val for row in grid for val in row]
    background = Counter(flat).most_common(1)[0][0]

    # Extract sparse markers from rows 1,3,5,... at columns 1,3,5,...
    # Markers should be isolated cells, not part of filled blocks
    markers = []  # List of (row_index, [values])
    for r in range(1, min(10, rows), 2):  # Check rows 1,3,5,7,9
        values = []
        for c in range(1, min(10, cols), 2):  # Check columns 1,3,5,7,9
            if grid[r][c] != background:
                # Check if this cell is isolated (marker) or part of a filled block
                left_is_bg = (c == 0 or grid[r][c-1] == background)
                right_is_bg = (c >= cols - 1 or grid[r][c+1] == background)
                if left_is_bg and right_is_bg:
                    values.append(grid[r][c])
        if values:
            markers.append((r, values))

    if not markers:
        return result

    # Find existing filled blocks
    visited = [[False] * cols for _ in range(rows)]
    blocks = []  # List of (start_row, start_col, height, width, value)

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != background:
                # Check if this is part of a filled block
                val = grid[r][c]
                # Try to find a rectangular block
                height = 1
                width = 1

                # Expand right
                while c + width < cols and grid[r][c + width] == val:
                    width += 1

                # Expand down
                is_block = width >= 2  # At least 2 wide
                if is_block:
                    while r + height < rows:
                        # Check if all cells in this row match
                        if all(grid[r + height][c + w] == val for w in range(width) if c + w < cols):
                            height += 1
                        else:
                            break

                    if height >= 2:  # At least 2 tall
                        blocks.append((r, c, height, width, val))
                        # Mark as visited
                        for dr in range(height):
                            for dc in range(width):
                                if r + dr < rows and c + dc < cols:
                                    visited[r + dr][c + dc] = True

    if not blocks:
        return result

    # Find unique block positions (columns) and dimensions
    block_cols = sorted(set(b[1] for b in blocks))
    block_height = blocks[0][2]
    block_width = blocks[0][3]

    # Match marker row to existing blocks
    block_values = [b[4] for b in blocks]
    block_value_set = set(block_values)

    matching_index = -1
    for i, (marker_row, marker_vals) in enumerate(markers):
        marker_set = set(marker_vals)
        if marker_set & block_value_set:  # Intersection
            if marker_set == block_value_set or block_value_set.issubset(marker_set):
                matching_index = i
                break

    if matching_index == -1:
        matching_index = 0

    # Get the row position of existing blocks
    existing_block_row = blocks[0][0]

    # Calculate spacing based on block height and which marker matches
    if matching_index == 0:  # First marker matches
        spacing = 2 * block_height - 1
    elif matching_index == len(markers) - 1:  # Last marker matches
        spacing = 2 * block_height + 1
    else:  # Middle marker matches
        spacing = 2 * block_height + 1

    # Place blocks for each marker
    for i, (marker_row, marker_vals) in enumerate(markers):
        # Calculate target row
        target_row = existing_block_row + (i - matching_index) * spacing

        if target_row < 0 or target_row + block_height > rows:
            continue

        # Place blocks at block_cols positions
        for block_idx, col_pos in enumerate(block_cols):
            if block_idx < len(marker_vals):
                val = marker_vals[block_idx]
                # Fill the block
                for dr in range(block_height):
                    for dc in range(block_width):
                        if target_row + dr < rows and col_pos + dc < cols:
                            result[target_row + dr][col_pos + dc] = val

        # Handle extra marker values (create new blocks)
        if len(marker_vals) > len(block_cols):
            # Calculate new column positions
            if len(block_cols) >= 2:
                col_spacing = block_cols[1] - block_cols[0]
            else:
                col_spacing = block_width + 1

            for extra_idx in range(len(block_cols), len(marker_vals)):
                new_col = block_cols[-1] + col_spacing * (extra_idx - len(block_cols) + 1)
                val = marker_vals[extra_idx]
                for dr in range(block_height):
                    for dc in range(block_width):
                        if target_row + dr < rows and new_col + dc < cols:
                            result[target_row + dr][new_col + dc] = val

    return result
