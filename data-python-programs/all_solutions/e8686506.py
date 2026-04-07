def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has background color and multiple non-background colored patterns
    2. Large hollow frames with same width/columns stack vertically
    3. Smaller centered patterns overlay in appropriate sections
    4. Scattered markers (≤4 cells) mark corners and junctions
    5. Patterns preserve their exact shapes from bounding boxes

    Procedure:
    1. Extract all colored regions with their bounding box patterns
    2. Identify stackable frames (same size, same columns)
    3. Overlay centered patterns into each frame section
    4. Place markers at structural positions (corners, junctions)
    """
    from collections import Counter

    # Find background color
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Extract cells for each color
    color_cells = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != bg:
                color = grid[r][c]
                if color not in color_cells:
                    color_cells[color] = []
                color_cells[color].append((r, c))

    if not color_cells:
        return [[bg]]

    # Build shape info
    shapes = {}
    for color, cells in color_cells.items():
        min_r, max_r = min(r for r, c in cells), max(r for r, c in cells)
        min_c, max_c = min(c for r, c in cells), max(c for r, c in cells)
        h, w = max_r - min_r + 1, max_c - min_c + 1

        # Extract pattern (with background as None for clarity)
        pattern = [[None] * w for _ in range(h)]
        for r, c in cells:
            pattern[r - min_r][c - min_c] = color

        shapes[color] = {
            'min_r': min_r, 'max_r': max_r, 'min_c': min_c, 'max_c': max_c,
            'h': h, 'w': w, 'count': len(cells), 'pattern': pattern
        }

    # Classify shapes: markers (≤4 cells) vs real shapes
    markers = [(c, s) for c, s in shapes.items() if s['count'] <= 4]
    real_shapes = [(c, s) for c, s in shapes.items() if s['count'] > 4]

    markers.sort()  # Consistent ordering
    real_shapes.sort(key=lambda x: (x[1]['count'], x[1]['min_r']), reverse=True)

    if not real_shapes:
        return [[bg]]

    # Check if top two shapes should stack (same width, same columns)
    if len(real_shapes) >= 2:
        c1, s1 = real_shapes[0]
        c2, s2 = real_shapes[1]

        if (s1['w'] == s2['w'] and s1['h'] == s2['h'] and
            s1['min_c'] == s2['min_c'] and s1['max_c'] == s2['max_c']):

            # These are stackable frames
            if s1['min_r'] < s2['min_r']:
                tc, ts, bc, bs = c1, s1, c2, s2
            else:
                tc, ts, bc, bs = c2, s2, c1, s1

            h, w = ts['h'] + bs['h'], ts['w']
            result = [[None] * w for _ in range(h)]

            # Place frame patterns (keep exact pattern, None stays None)
            for r in range(ts['h']):
                for c in range(w):
                    result[r][c] = ts['pattern'][r][c]

            for r in range(bs['h']):
                for c in range(w):
                    result[r + ts['h']][c] = bs['pattern'][r][c]

            # Overlay other real shapes into appropriate sections
            used = {tc, bc}
            for color, shape in real_shapes:
                if color in used:
                    continue

                # Determine which section based on row position
                if shape['max_r'] <= ts['max_r']:
                    # Overlay in top section, starting from row 1
                    sr = 1
                    sc = (w - shape['w']) // 2
                    for r in range(shape['h']):
                        for c in range(shape['w']):
                            if shape['pattern'][r][c] is not None:
                                rr, cc = sr + r, sc + c
                                if 0 <= rr < ts['h'] and 0 <= cc < w:
                                    result[rr][cc] = shape['pattern'][r][c]
                else:
                    # Overlay in bottom section, centered
                    sr = ts['h'] + (bs['h'] - shape['h']) // 2
                    sc = (w - shape['w']) // 2
                    for r in range(shape['h']):
                        for c in range(shape['w']):
                            if shape['pattern'][r][c] is not None:
                                rr, cc = sr + r, sc + c
                                if ts['h'] <= rr < h and 0 <= cc < w:
                                    result[rr][cc] = shape['pattern'][r][c]
                used.add(color)

            # Place markers - corners first, then junctions
            if len(markers) >= 1:
                mc1, ms1 = markers[0]
                result[0][0] = result[0][w-1] = mc1
                result[h-1][0] = result[h-1][w-1] = mc1

            if len(markers) >= 2:
                mc2, ms2 = markers[1]
                junc = ts['h']
                result[junc-1][0] = result[junc-1][w-1] = mc2
                result[junc][0] = result[junc][w-1] = mc2

            # Fill remaining Nones with frame colors
            for r in range(h):
                for c in range(w):
                    if result[r][c] is None:
                        if r < ts['h']:
                            result[r][c] = tc
                        else:
                            result[r][c] = bc

            return result

    # Single primary shape case
    pc, ps = real_shapes[0]
    h, w = ps['h'], ps['w']
    result = [row[:] for row in ps['pattern']]

    # Overlay other shapes
    used = {pc}
    for i, (color, shape) in enumerate(real_shapes[1:]):
        # Position based on index: first in upper area, second in lower area
        if i == 0:
            sr = 1  # Start from row 1
        else:
            sr = h - shape['h']  # Bottom area
        sc = (w - shape['w']) // 2
        for r in range(shape['h']):
            for c in range(shape['w']):
                if shape['pattern'][r][c] is not None:
                    rr, cc = sr + r, sc + c
                    if 0 <= rr < h and 0 <= cc < w:
                        result[rr][cc] = shape['pattern'][r][c]

    # Place markers
    junction_row = None
    if len(markers) >= 1:
        mc, ms = markers[0]
        result[0][w//2] = mc
        # Find junction rows (where None transitions) for sides
        for r in range(1, h-1):
            if result[r][0] is None or result[r][w-1] is None:
                result[r][0] = mc
                result[r][w-1] = mc
                if junction_row is None:
                    junction_row = r

    if len(markers) >= 2 and junction_row is not None:
        mc2, ms2 = markers[1]
        # Place second marker at junction center (vertical line)
        result[junction_row][w//2] = mc2
        if junction_row + 1 < h:
            result[junction_row + 1][w//2] = mc2

    # Fill remaining Nones with primary color
    for r in range(h):
        for c in range(w):
            if result[r][c] is None:
                result[r][c] = pc

    return result
