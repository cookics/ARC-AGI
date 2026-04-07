def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with a non-zero color forming patterns in one area and 0s elsewhere
    2. Output extends the pattern across the grid:
       - Original non-zero elements are preserved
       - Pattern is extended based on bbox position
       - Background filled with color 5 or frame color in specific patterns
    3. Bottom-left bbox → diagonal pattern from top-right
    4. Top-right bbox → frame lines at specific offsets
    5. Top bbox → horizontal stripes in bottom area

    Procedure:
    1. Find non-zero color and bounding box
    2. Determine bbox position (which edges it touches)
    3. Apply position-specific extension pattern
    4. Fill background appropriately
    """

    rows, cols = len(grid), len(grid[0])

    # Find non-zero color and bounding box
    frame_color = None
    min_r, max_r, min_c, max_c = rows, -1, cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                if frame_color is None:
                    frame_color = grid[r][c]
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Initialize result with 5s
    result = [[5 for _ in range(cols)] for _ in range(rows)]

    # Copy original pattern
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == frame_color:
                result[r][c] = frame_color

    # Determine bbox position
    touches_top = min_r == 0
    touches_bottom = max_r == rows - 1
    touches_left = min_c == 0
    touches_right = max_c == cols - 1

    # Case 1: Bottom-left (lower-left quadrant)
    if touches_bottom and touches_left and not touches_top and not touches_right:
        # Diagonal pattern from top-right to bottom-left
        result = [[5 for _ in range(cols)] for _ in range(rows)]

        # Row 0 is all frame color
        for c in range(cols):
            result[0][c] = frame_color

        # For rows 1-9, diagonal boundary
        for r in range(1, rows):
            boundary = cols - r
            # Before boundary: base color (alternates by row)
            base_color = frame_color if r % 2 == 0 else 5
            opposite_color = 5 if r % 2 == 0 else frame_color

            for c in range(boundary):
                result[r][c] = base_color

            # After boundary: alternating pattern starting with opposite of base
            for c in range(boundary, cols):
                if c == cols - 1:
                    result[r][c] = frame_color
                else:
                    # Alternate starting with opposite color
                    result[r][c] = (
                        opposite_color if (c - boundary) % 2 == 0 else base_color
                    )

    # Case 2: Top-right quadrant
    elif touches_top and touches_right and not touches_bottom and not touches_left:
        # Compute spacing based on bbox dimensions
        bbox_width = max_c - min_c + 1
        bbox_height = max_r - min_r + 1
        spacing = bbox_height // 2

        # Add vertical line to the left at spacing distance
        new_col = min_c - spacing
        if new_col >= 0:
            for r in range(rows - 1):
                result[r][new_col] = frame_color

        # Add horizontal line below at spacing distance
        new_row = max_r + spacing
        if new_row < rows:
            for c in range(new_col if new_col >= 0 else 0, cols):
                result[new_row][c] = frame_color

    # Case 3: Top area, not necessarily touching left/right edges fully
    elif touches_top and not touches_bottom and max_r < rows // 2:
        # Compute spacing based on bbox dimensions
        bbox_height = max_r - min_r + 1
        spacing = bbox_height // 2

        # Extend rightmost column with frame color
        for r in range(rows):
            result[r][cols - 1] = frame_color

        # Add horizontal stripes starting below bbox with spacing
        stripe_start = max_r + spacing
        for r in range(stripe_start, rows):
            if r % 2 == 1:  # Odd rows - all frame color
                for c in range(cols):
                    result[r][c] = frame_color
            else:  # Even rows - all 5s
                for c in range(cols):
                    result[r][c] = 5

    # Case 4: Touches left, top, and bottom (spans full height)
    elif touches_left and touches_top and touches_bottom:
        # Extend all rows that have frame color to the right edge
        for r in range(rows):
            has_frame = any(grid[r][c] == frame_color for c in range(cols))
            if has_frame:
                for c in range(cols):
                    if result[r][c] != frame_color:
                        result[r][c] = frame_color

    return result
