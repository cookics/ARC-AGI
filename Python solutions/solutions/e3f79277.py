def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 6×6 grid with background color (7) and an L-shaped pattern in a corner
    2. Output is a 16×16 grid
    3. The pattern is scaled 2× and creates a triangular fill with:
       - Two edges fully filled (meeting at the corner)
       - A diagonal connecting them
    4. Example 1: bottom-right corner pattern with anti-diagonal
       Example 2: bottom-left corner pattern with main diagonal

    Procedure:
    1. Find bounding box of non-background cells and identify corner location
    2. Scale the bounding box by 2× and position it appropriately in output
    3. Fill the triangular pattern with edges and diagonal
    """

    # Find bounding box of non-7 cells
    min_r, max_r = 6, -1
    min_c, max_c = 6, -1
    bg_color = 7
    pattern_color = None

    for r in range(6):
        for c in range(6):
            if grid[r][c] != bg_color:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
                pattern_color = grid[r][c]

    # Calculate bounding box dimensions
    h_input = max_r - min_r + 1
    w_input = max_c - min_c + 1

    # Determine corner by checking which corner of bbox is closest to grid corner
    bbox_corners = [
        (min_r, min_c, 'TL'), (min_r, max_c, 'TR'),
        (max_r, min_c, 'BL'), (max_r, max_c, 'BR'),
    ]
    grid_corners = [
        (0, 0, 'TL'), (0, 5, 'TR'), (5, 0, 'BL'), (5, 5, 'BR'),
    ]

    min_dist = float('inf')
    corner = None
    for br, bc, b_label in bbox_corners:
        for gr, gc, g_label in grid_corners:
            if b_label == g_label:
                dist = abs(br - gr) + abs(bc - gc)
                if dist < min_dist:
                    min_dist = dist
                    corner = g_label

    # Scale dimensions by 2×
    h_output = h_input * 2
    w_output = w_input * 2

    # Create output grid
    result = [[bg_color] * 16 for _ in range(16)]

    # Calculate position and fill pattern based on corner
    if corner == 'BR':  # bottom-right
        dist_from_bottom = 5 - max_r
        dist_from_right = 5 - max_c
        r0 = 15 - h_output + 1 - dist_from_bottom * 2
        c0 = 15 - w_output + 1 - dist_from_right * 2

        # Right edge, bottom edge, anti-diagonal
        for r in range(r0, r0 + h_output):
            result[r][c0 + w_output - 1] = pattern_color
        for c in range(c0, c0 + w_output):
            result[r0 + h_output - 1][c] = pattern_color
        for i in range(1, h_output - 1):
            result[r0 + i][c0 + w_output - 1 - i] = pattern_color

    elif corner == 'BL':  # bottom-left
        dist_from_bottom = 5 - max_r
        r0 = 15 - h_output + 1 - dist_from_bottom * 2
        c0 = min_c * 2

        # Left edge, bottom edge, main diagonal
        for r in range(r0, r0 + h_output):
            result[r][c0] = pattern_color
        for c in range(c0, c0 + w_output):
            result[r0 + h_output - 1][c] = pattern_color
        for i in range(1, h_output - 1):
            result[r0 + i][c0 + i] = pattern_color

    elif corner == 'TR':  # top-right
        dist_from_right = 5 - max_c
        r0 = min_r * 2
        c0 = 15 - w_output + 1 - dist_from_right * 2

        # Right edge, top edge, anti-diagonal
        for r in range(r0, r0 + h_output):
            result[r][c0 + w_output - 1] = pattern_color
        for c in range(c0, c0 + w_output):
            result[r0][c] = pattern_color
        for i in range(1, h_output - 1):
            result[r0 + i][c0 + w_output - 1 - i] = pattern_color

    elif corner == 'TL':  # top-left
        r0 = min_r * 2
        c0 = min_c * 2

        # Left edge, top edge, main diagonal
        for r in range(r0, r0 + h_output):
            result[r][c0] = pattern_color
        for c in range(c0, c0 + w_output):
            result[r0][c] = pattern_color
        for i in range(1, h_output - 1):
            result[r0 + i][c0 + i] = pattern_color

    return result
