def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains hollow rectangles made of 3s (boundary is 3, interior is 0)
    2. Rectangles with overlapping interior regions get connected
    3. Connections involve: filling gaps between rectangles, creating openings in edges, adding extensions

    Procedure:
    1. Detect all hollow rectangles
    2. Find pairs of rectangles with overlapping interiors (horizontally or vertically aligned)
    3. For horizontal alignment: fill gap between rectangles
    4. For vertical alignment: create openings and extensions, fill partial gap
    """

    def is_hollow_rect(grid, r1, c1, r2, c2):
        """Check if region forms a hollow rectangle"""
        # Check all edges are 3s
        for c in range(c1, c2 + 1):
            if grid[r1][c] != 3 or grid[r2][c] != 3:
                return False
        for r in range(r1, r2 + 1):
            if grid[r][c1] != 3 or grid[r][c2] != 3:
                return False

        # Check interior is all 0s
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if grid[r][c] != 0:
                    return False
        return True

    def create_vertical_connection(grid, upper_rect, lower_rect, overlap_c_start, overlap_c_end):
        """Create vertical connection with openings and extensions"""
        r1_top, c1_left, r1_bot, c1_right = upper_rect
        r2_top, c2_left, r2_bot, c2_right = lower_rect

        r1_int_top, r1_int_bot = r1_top + 1, r1_bot - 1
        r2_int_top, r2_int_bot = r2_top + 1, r2_bot - 1

        # Calculate opening in upper rectangle's bottom edge
        overlap_width = overlap_c_end - overlap_c_start + 1
        opening_width = max(1, overlap_width - 2)
        opening_start = overlap_c_start + (overlap_width - opening_width) // 2
        opening_end = opening_start + opening_width - 1

        # Create opening in upper rectangle's bottom edge
        for c in range(opening_start, opening_end + 1):
            grid[r1_bot][c] = 0

        # Add extensions below the opening
        if r1_bot + 1 < len(grid):
            grid[r1_bot + 1][overlap_c_start] = 3
            grid[r1_bot + 1][overlap_c_end] = 3

        # Create opening in lower rectangle's top edge
        for c in range(opening_start, opening_end + 1):
            grid[r2_top][c] = 0

        # Create side openings at overlap boundaries for interior rows
        # Open right edge of upper rect at overlap_c_end if needed
        if overlap_c_end == c1_right:
            for r in range(r1_int_top, r1_int_bot + 1):
                if grid[r][c1_right] == 3:
                    grid[r][c1_right] = 0

        # Open left edge of upper rect at overlap_c_start if needed
        if overlap_c_start == c1_left:
            for r in range(r1_int_top, r1_int_bot + 1):
                if grid[r][c1_left] == 3:
                    grid[r][c1_left] = 0

    def has_rect_in_between_horiz(rect1, rect2, all_rects):
        """Check if there's another rectangle between rect1 and rect2 horizontally"""
        r1_top, c1_left, r1_bot, c1_right = rect1
        r2_top, c2_left, r2_bot, c2_right = rect2

        # Ensure rect1 is to the left of rect2
        if c1_right > c2_left:
            rect1, rect2 = rect2, rect1
            r1_top, c1_left, r1_bot, c1_right = rect1
            r2_top, c2_left, r2_bot, c2_right = rect2

        # Check if any other rectangle is between them
        for rect in all_rects:
            if rect == rect1 or rect == rect2:
                continue
            r_top, c_left, r_bot, c_right = rect
            # Check if this rectangle is between rect1 and rect2 horizontally
            # and overlaps in rows
            if c_left > c1_right and c_right < c2_left:
                if not (r_bot < min(r1_top, r2_top) or r_top > max(r1_bot, r2_bot)):
                    return True
        return False

    def connect_rectangles(grid, rect1, rect2, all_rects):
        """Connect two rectangles if they are aligned and directly adjacent"""
        r1_top, c1_left, r1_bot, c1_right = rect1
        r2_top, c2_left, r2_bot, c2_right = rect2

        # Interior ranges
        r1_int_top, r1_int_bot = r1_top + 1, r1_bot - 1
        r1_int_left, r1_int_right = c1_left + 1, c1_right - 1
        r2_int_top, r2_int_bot = r2_top + 1, r2_bot - 1
        r2_int_left, r2_int_right = c2_left + 1, c2_right - 1

        # Check horizontal alignment (interior rows overlap)
        overlap_r_start = max(r1_int_top, r2_int_top)
        overlap_r_end = min(r1_int_bot, r2_int_bot)

        if overlap_r_start <= overlap_r_end:
            # Rectangles are horizontally aligned
            if c1_right < c2_left or c2_right < c1_left:
                # Check if there's no rectangle in between
                if not has_rect_in_between_horiz(rect1, rect2, all_rects):
                    # Determine which rows to fill
                    rows_to_fill = []
                    if overlap_r_end - overlap_r_start <= 1:
                        # Fill all overlapping rows if there are only 1-2 rows
                        rows_to_fill = list(range(overlap_r_start, overlap_r_end + 1))
                    else:
                        # Fill only first and last overlapping rows if there are more
                        rows_to_fill = [overlap_r_start, overlap_r_end]

                    if c1_right < c2_left:
                        # rect1 is left of rect2 - fill gap
                        for r in rows_to_fill:
                            for c in range(c1_right + 1, c2_left):
                                grid[r][c] = 3
                    elif c2_right < c1_left:
                        # rect2 is left of rect1 - fill gap
                        for r in rows_to_fill:
                            for c in range(c2_right + 1, c1_left):
                                grid[r][c] = 3

        # Check vertical alignment (interior cols overlap)
        overlap_c_start = max(r1_int_left, r2_int_left)
        overlap_c_end = min(r1_int_right, r2_int_right)

        if overlap_c_start <= overlap_c_end:
            # Rectangles are vertically aligned
            if r1_bot < r2_top:
                # rect1 is above rect2
                create_vertical_connection(grid, rect1, rect2, overlap_c_start, overlap_c_end)
            elif r2_bot < r1_top:
                # rect2 is above rect1
                create_vertical_connection(grid, rect2, rect1, overlap_c_start, overlap_c_end)

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Detect all hollow rectangles
    rectangles = []
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1][c1] != 3:
                continue
            for r2 in range(r1 + 2, rows):
                for c2 in range(c1 + 2, cols):
                    if is_hollow_rect(grid, r1, c1, r2, c2):
                        rectangles.append((r1, c1, r2, c2))

    # Remove duplicate rectangles
    rectangles = list(set(rectangles))

    # Process connections between rectangles
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            r1, r2 = rectangles[i], rectangles[j]
            connect_rectangles(result, r1, r2, rectangles)

    return result
