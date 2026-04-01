def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a large rectangle filled with a single color (fill_color)
    2. Input has small pattern blocks (2-3 blocks) containing mixed colors and 8s
    3. Output is the size of the large rectangle, filled with fill_color
    4. One pattern (with non-fill color) is selected, 8s extracted, and placed at multiple locations

    Procedure:
    1. Find the large filled rectangle and its fill color
    2. Find small pattern blocks in the top area
    3. Select the pattern with a color different from fill_color (excluding 0 and 8)
    4. Extract 8 positions from the selected pattern
    5. Place the 8 pattern at specific locations on the canvas:
       - Top-center: (0, centered_col)
       - Middle-left: (pattern_height, 0)
       - Middle-right: (pattern_height, canvas_width - pattern_width)
       - Bottom-center (if space): (2*pattern_height, centered_col)
    """

    rows, cols = len(grid), len(grid[0])

    # Find the large filled rectangle
    fill_color = None
    large_rect = None

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 8:
                # Check if this starts a large filled region
                color = grid[r][c]
                r1, c1 = r, c
                r2, c2 = r, c

                # Expand to find the rectangle
                while r2 + 1 < rows and grid[r2 + 1][c] == color:
                    r2 += 1
                while c2 + 1 < cols and grid[r][c2 + 1] == color:
                    c2 += 1

                # Check if it's a large filled rectangle
                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                is_filled = True
                for rr in range(r1, r2 + 1):
                    for cc in range(c1, c2 + 1):
                        if grid[rr][cc] != color:
                            is_filled = False
                            break
                    if not is_filled:
                        break

                if is_filled and area > 20:  # Large enough rectangle
                    if large_rect is None or area > large_rect[4]:
                        large_rect = (r1, c1, r2, c2, area)
                        fill_color = color

    if large_rect is None:
        return grid

    r1, c1, r2, c2, _ = large_rect
    canvas_height = r2 - r1 + 1
    canvas_width = c2 - c1 + 1

    # Find pattern blocks (above the large rectangle)
    pattern_blocks = []
    visited = [[False] * cols for _ in range(rows)]

    for r in range(r1):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                # Find the bounding box of this pattern block
                min_r, max_r = r, r
                min_c, max_c = c, c

                # BFS to find the connected region
                queue = [(r, c)]
                visited[r][c] = True
                cells = [(r, c)]

                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                            if grid[nr][nc] != 0:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
                                cells.append((nr, nc))
                                min_r = min(min_r, nr)
                                max_r = max(max_r, nr)
                                min_c = min(min_c, nc)
                                max_c = max(max_c, nc)

                if max_r - min_r >= 2 and max_c - min_c >= 2:  # Valid pattern block
                    pattern_blocks.append((min_r, min_c, max_r, max_c))

    # Select the pattern with a non-fill color
    selected_pattern = None
    for pr1, pc1, pr2, pc2 in pattern_blocks:
        has_non_fill = False
        for r in range(pr1, pr2 + 1):
            for c in range(pc1, pc2 + 1):
                if grid[r][c] != 0 and grid[r][c] != 8 and grid[r][c] != fill_color:
                    has_non_fill = True
                    break
            if has_non_fill:
                break

        if has_non_fill:
            selected_pattern = (pr1, pc1, pr2, pc2)
            break

    if selected_pattern is None:
        # Fallback: use the first pattern
        if pattern_blocks:
            selected_pattern = pattern_blocks[0]
        else:
            # No patterns found, return filled canvas
            return [[fill_color] * canvas_width for _ in range(canvas_height)]

    # Extract 8s from the selected pattern
    pr1, pc1, pr2, pc2 = selected_pattern
    pattern_height = pr2 - pr1 + 1
    pattern_width = pc2 - pc1 + 1

    eights_pattern = []
    for r in range(pr1, pr2 + 1):
        for c in range(pc1, pc2 + 1):
            if grid[r][c] == 8:
                eights_pattern.append((r - pr1, c - pc1))

    # Create output canvas
    result = [[fill_color] * canvas_width for _ in range(canvas_height)]

    # Place the pattern at multiple locations
    def place_pattern(start_row, start_col):
        for dr, dc in eights_pattern:
            rr = start_row + dr
            cc = start_col + dc
            if 0 <= rr < canvas_height and 0 <= cc < canvas_width:
                result[rr][cc] = 8

    # Top-center (different formula for even vs odd canvas width)
    if canvas_width % 2 == 0:
        center_col = canvas_width // 2 - pattern_width
    else:
        center_col = (canvas_width - pattern_width) // 2
    place_pattern(0, center_col)

    # Middle-left
    if pattern_height < canvas_height:
        place_pattern(pattern_height, 0)

    # Middle-right
    if pattern_height < canvas_height:
        place_pattern(pattern_height, canvas_width - pattern_width)

    # Bottom-center (if there's space)
    if canvas_height > 2 * pattern_height:
        place_pattern(2 * pattern_height, center_col)

    return result
