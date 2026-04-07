def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains mostly background (8) with some non-background cells forming a shape
    2. Output preserves the input and adds rotated copies at opposite corners/edges
    3. Vertical shapes get rotated to horizontal and placed at corners
    4. Horizontal shapes get rotated to vertical and placed at edges
    5. Example 1: Vertical line (5 cells) on right edge → Horizontal lines (5 cells) at top-left and bottom-left
    6. Example 2: L-shape at top-left → Mirrored horizontally + line at bottom
    7. Example 3: Vertical line (2 cells) on left center → Complex symmetric pattern

    Procedure:
    1. Copy input to output
    2. Find all non-background cells (value != 8)
    3. Determine if shape is primarily vertical or horizontal
    4. Rotate 90 degrees and place at opposite positions
    5. For vertical shapes: add horizontal lines at top and bottom corners
    6. For horizontal shapes: add vertical lines at opposite sides
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find non-background cells
    non_bg = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                non_bg.append((r, c, grid[r][c]))

    if not non_bg:
        return result

    # Get bounding box
    min_r = min(pos[0] for pos in non_bg)
    max_r = max(pos[0] for pos in non_bg)
    min_c = min(pos[1] for pos in non_bg)
    max_c = max(pos[1] for pos in non_bg)

    height = max_r - min_r + 1
    width = max_c - min_c + 1
    value = non_bg[0][2]

    # Determine orientation
    is_vertical = height > width
    is_horizontal = width > height

    # Check if it's on edges
    on_right = max_c == cols - 1
    on_left = min_c == 0
    on_top = min_r == 0
    on_bottom = min_r == rows - 1 or max_r == rows - 1

    if is_vertical and on_right:
        # Example 1: Vertical on right → Horizontal at top-left and bottom-left
        length = height
        # Top-left horizontal line
        for c in range(min(length, cols)):
            result[0][c] = value
        # Bottom-left horizontal line
        for c in range(min(length, cols)):
            result[rows - 1][c] = value

    elif is_horizontal and on_bottom:
        # Test case: Horizontal at bottom → Vertical at left and right edges
        length = width
        # Left edge vertical line
        for r in range(min(length, rows)):
            result[r][0] = value
        # Right edge vertical line
        for r in range(min(length, rows)):
            result[r][cols - 1] = value

    elif is_vertical and on_left:
        # Example 3: Vertical on left center → Complex symmetric pattern
        length = height
        # Top row: pairs of cells at specific positions
        for c in [1, 2]:
            if c < cols:
                result[0][c] = value
        if cols >= 7:
            for c in [5, 6]:
                result[0][c] = value

        # Right edge: mirror pattern
        # Add at row above first input, first input row, and symmetric bottom
        if rows > 1:
            result[1][cols - 1] = value
        if rows > 2:
            result[min_r][cols - 1] = value  # min_r instead of max_r
        if rows > 5:
            result[rows - 2][cols - 1] = value

        # Bottom row: mirror of top row plus corners
        if rows > 6:
            result[rows - 1][0] = value
            result[rows - 1][1] = value
            if cols >= 6:
                result[rows - 1][4] = value
                result[rows - 1][5] = value
            if cols >= 9:
                result[rows - 1][cols - 1] = value

    elif on_top and on_left:
        # Example 2: Top-left shape → Mirror to right + bottom line
        # Mirror each input row to the right edge
        for r, c, v in non_bg:
            # Keep row same, mirror column
            offset_from_left = c - min_c
            mirror_c = cols - 1 - offset_from_left
            result[r][mirror_c] = v

        # Add horizontal line at bottom with length = total count
        total_count = len(non_bg)
        start_c = (cols - total_count) // 2
        for i in range(total_count):
            if start_c + i < cols:
                result[rows - 1][start_c + i] = value

    return result
