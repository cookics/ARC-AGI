def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 27x27 grid divided by lines of 1s at row 13 and column 13 into 4 quadrants
    2. Background is 4, separator is 1
    3. Patterns from top-left get reflected to other quadrants
    4. Anchor blocks (same color in source and target) determine positioning and scaling
    5. Each color block is scaled proportionally based on anchor size ratio

    Procedure:
    1. Find separator lines (row and column of all 1s)
    2. Extract all non-background cells from each quadrant
    3. For each quadrant, reflect top-left pattern with proper transformation
    4. Use anchor color (common between source and target) to align and scale
    5. Place scaled color blocks at aligned positions
    """
    import copy
    from collections import defaultdict

    result = copy.deepcopy(grid)
    h, w = len(grid), len(grid[0])

    # Find separators
    sep_row = next(r for r in range(h) if all(grid[r][c] == 1 for c in range(w)))
    sep_col = next(c for c in range(w) if all(grid[r][c] == 1 for r in range(h)))

    def get_cells(r_start, r_end, c_start, c_end):
        """Get (local_row, local_col, color) for non-4 cells"""
        cells = []
        for r in range(r_start, r_end + 1):
            for c in range(c_start, c_end + 1):
                if grid[r][c] != 4:
                    cells.append((r - r_start, c - c_start, grid[r][c]))
        return cells

    def get_bbox(cells, color):
        """Get (min_r, max_r, min_c, max_c) for a color"""
        pos = [(r, c) for r, c, col in cells if col == color]
        if not pos:
            return None
        return (min(r for r, c in pos), max(r for r, c in pos),
                min(c for r, c in pos), max(c for r, c in pos))

    # Extract quadrant data
    tl_cells = get_cells(0, sep_row - 1, 0, sep_col - 1)
    tl_colors = set(c for _, _, c in tl_cells)

    def reflect_h(cells, max_c):
        """Reflect horizontally"""
        return [(r, max_c - c, col) for r, c, col in cells]

    def reflect_v(cells, max_r):
        """Reflect vertically"""
        return [(max_r - r, c, col) for r, c, col in cells]

    def apply_pattern(src_cells, src_colors, tgt_cells, tgt_colors,
                     refl_cells, quad_r_off, quad_c_off, quad_h, quad_w):
        """Apply reflected pattern to target quadrant"""
        common = src_colors & tgt_colors

        if not common:
            # No anchor - just place reflected pattern
            for r, c, col in refl_cells:
                if 0 <= r < quad_h and 0 <= c < quad_w:
                    result[r + quad_r_off][c + quad_c_off] = col
            return

        anchor = min(common)

        # Get bounding boxes
        refl_bbox = get_bbox(refl_cells, anchor)
        tgt_bbox = get_bbox(tgt_cells, anchor)

        if not refl_bbox or not tgt_bbox:
            return

        # Calculate scale based on anchor sizes
        refl_h = refl_bbox[1] - refl_bbox[0] + 1
        refl_w = refl_bbox[3] - refl_bbox[2] + 1
        tgt_h = tgt_bbox[1] - tgt_bbox[0] + 1
        tgt_w = tgt_bbox[3] - tgt_bbox[2] + 1

        scale_r = tgt_h / refl_h
        scale_c = tgt_w / refl_w

        # Calculate offset to align anchors
        offset_r = tgt_bbox[0] - refl_bbox[0]
        offset_c = tgt_bbox[2] - refl_bbox[2]

        # Group cells by color
        by_color = defaultdict(list)
        for r, c, col in refl_cells:
            by_color[col].append((r, c))

        # Place each color block with scaling
        for col, positions in by_color.items():
            bbox = get_bbox(refl_cells, col)
            if not bbox:
                continue

            min_r, max_r, min_c, max_c = bbox

            # Scaled dimensions
            new_h = max(1, round((max_r - min_r + 1) * scale_r))
            new_w = max(1, round((max_c - min_c + 1) * scale_c))

            # Position relative to anchor
            rel_r = min_r - refl_bbox[0]
            rel_c = min_c - refl_bbox[2]

            # Scaled position
            scaled_r = round(rel_r * scale_r)
            scaled_c = round(rel_c * scale_c)

            # Absolute position in quadrant
            abs_r = tgt_bbox[0] + scaled_r
            abs_c = tgt_bbox[2] + scaled_c

            # Place block
            for dr in range(new_h):
                for dc in range(new_w):
                    rr = abs_r + dr
                    cc = abs_c + dc
                    if 0 <= rr < quad_h and 0 <= cc < quad_w:
                        result[quad_r_off + rr][quad_c_off + cc] = col

    # Process each quadrant
    quad_h = sep_row
    quad_w = sep_col

    # Top-right
    tr_cells = get_cells(0, sep_row - 1, sep_col + 1, w - 1)
    tr_colors = set(c for _, _, c in tr_cells)
    refl_tr = reflect_h(tl_cells, quad_w - 1)
    apply_pattern(tl_cells, tl_colors, tr_cells, tr_colors, refl_tr,
                 0, sep_col + 1, quad_h, quad_w)

    # Bottom-left
    bl_cells = get_cells(sep_row + 1, h - 1, 0, sep_col - 1)
    bl_colors = set(c for _, _, c in bl_cells)
    refl_bl = reflect_v(tl_cells, quad_h - 1)
    apply_pattern(tl_cells, tl_colors, bl_cells, bl_colors, refl_bl,
                 sep_row + 1, 0, quad_h, quad_w)

    # Bottom-right
    br_cells = get_cells(sep_row + 1, h - 1, sep_col + 1, w - 1)
    br_colors = set(c for _, _, c in br_cells)
    refl_br = reflect_v(reflect_h(tl_cells, quad_w - 1), quad_h - 1)
    apply_pattern(tl_cells, tl_colors, br_cells, br_colors, refl_br,
                 sep_row + 1, sep_col + 1, quad_h, quad_w)

    return result
