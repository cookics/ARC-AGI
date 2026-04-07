def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Connects two rectangular blocks of 1s with a frame structure using pattern elements found around the first block.

    Procedure:
    1. Find two blocks of 1s
    2. Extract pattern elements from around first block
    3. Create frame connecting blocks with extracted patterns
    """

    rows, cols = len(grid), len(grid[0])

    # Find background color (most frequent)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    background = max(color_count, key=color_count.get)

    # Find all blocks of 1s
    blocks = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] == 1:
                # Find extent of this block
                min_r, max_r, min_c, max_c = r, r, c, c

                # Expand horizontally
                while max_c + 1 < cols and grid[r][max_c + 1] == 1:
                    max_c += 1

                # Expand vertically
                while max_r + 1 < rows and grid[max_r + 1][c] == 1:
                    max_r += 1

                # Mark all cells in block as visited
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        visited.add((rr, cc))

                blocks.append((min_r, max_r, min_c, max_c))

    if len(blocks) != 2:
        return [row[:] for row in grid]

    # Sort blocks by top-left position
    blocks.sort()
    (r1, r2, c1, c2), (r3, r4, c3, c4) = blocks

    # Copy original grid
    result = [row[:] for row in grid]

    # Extract pattern elements from around first block
    h_color = None  # Right of first block
    pattern_color = None  # Further right
    v_color = None  # Below first block

    # Look to the right
    if c2 + 1 < cols and grid[r1][c2 + 1] != background:
        h_color = grid[r1][c2 + 1]

    # Look further right
    if c2 + 3 < cols and grid[r1][c2 + 3] != background:
        pattern_color = grid[r1][c2 + 3]

    # Look below
    if r2 + 1 < rows and grid[r2 + 1][c1] != background:
        v_color = grid[r2 + 1][c1]

    # Alternative: look below for pattern if not found horizontally
    if pattern_color is None and r2 + 3 < rows and grid[r2 + 3][c1] != background:
        pattern_color = grid[r2 + 3][c1]

    if h_color is None or pattern_color is None or v_color is None:
        return result

    # Frame boundaries
    top_frame = r1 - 1
    bottom_frame = r4 + 1
    left_frame = c1 - 1
    right_frame = c4 + 1

    # Create horizontal frame lines
    if 0 <= top_frame < rows:
        for c in range(c2 + 1, c3):
            if 0 <= c < cols:
                result[top_frame][c] = h_color

    if 0 <= bottom_frame < rows:
        for c in range(c2 + 1, c3):
            if 0 <= c < cols:
                result[bottom_frame][c] = h_color

    # Fill anchor rows with repeating pattern
    anchor_pattern = [h_color, background, pattern_color, background]

    # First anchor block
    for r in range(r1, r2 + 1):
        for i, c in enumerate(range(c2 + 1, c3)):
            if 0 <= c < cols:
                result[r][c] = anchor_pattern[i % 4]

    # Second anchor block
    for r in range(r3, r4 + 1):
        for i, c in enumerate(range(c2 + 1, c3)):
            if 0 <= c < cols:
                result[r][c] = anchor_pattern[i % 4]

    # Fill interior between blocks
    for r in range(r2 + 1, r3):
        row_offset = r - r2 - 1

        if row_offset % 4 == 0:
            # Full row of v_color
            for c in range(left_frame, right_frame + 1):
                if 0 <= c < cols:
                    result[r][c] = v_color
        elif row_offset % 4 == 2:
            # Pattern row with vertical borders
            for c in range(c1, c4 + 1):
                if 0 <= c < cols:
                    result[r][c] = pattern_color
            # Vertical borders
            if 0 <= left_frame < cols:
                result[r][left_frame] = v_color
            if 0 <= right_frame < cols:
                result[r][right_frame] = v_color
        else:
            # Just vertical borders
            if 0 <= left_frame < cols:
                result[r][left_frame] = v_color
            if 0 <= right_frame < cols:
                result[r][right_frame] = v_color

    return result
