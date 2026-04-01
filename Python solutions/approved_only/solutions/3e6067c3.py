def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has rectangular regions with colored centers
    2. Bottom instruction row contains a sequence of colors
    3. Each element in the sequence refers to a specific rectangle (Nth occurrence of that color)
    4. For consecutive pairs of rectangles in the sequence:
       - If horizontally aligned: extend first color toward second
       - If vertically aligned: place first color in vertical gap

    Procedure:
    1. Find instruction row and extract color sequence
    2. Find all rectangles and sort in reading order
    3. Map sequence to specific rectangles
    4. Connect consecutive rectangle pairs
    """

    import copy

    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find instruction row
    key_row = None
    background = None

    for i in range(rows - 1, -1, -1):
        row = grid[i]
        if len(set(row)) >= 3:
            most_common = max(set(row), key=row.count)
            if row.count(most_common) > len(row) // 2:
                is_alt = True
                for j in range(0, min(20, len(row)), 2):
                    if j < len(row) and row[j] != most_common:
                        is_alt = False
                        break
                if is_alt:
                    key_row = row
                    background = most_common
                    break

    if not key_row:
        return result

    # Extract sequence
    key_colors = []
    for i in range(1, len(key_row), 2):
        if i < len(key_row) and key_row[i] != background:
            key_colors.append(key_row[i])

    # Find all rectangles
    all_rects = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited or grid[r][c] == background:
                continue

            color = grid[r][c]
            if color not in key_colors:
                continue

            # BFS to find rectangle
            rect_cells = []
            queue = [(r, c)]
            rect_visited = set()

            while queue:
                cr, cc = queue.pop(0)
                if (cr, cc) in rect_visited:
                    continue
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if grid[cr][cc] != color:
                    continue

                rect_visited.add((cr, cc))
                visited.add((cr, cc))
                rect_cells.append((cr, cc))

                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) not in rect_visited:
                        queue.append((nr, nc))

            if rect_cells:
                min_r = min(p[0] for p in rect_cells)
                max_r = max(p[0] for p in rect_cells)
                min_c = min(p[1] for p in rect_cells)
                max_c = max(p[1] for p in rect_cells)
                all_rects.append({
                    'color': color,
                    'cells': rect_cells,
                    'min_r': min_r,
                    'max_r': max_r,
                    'min_c': min_c,
                    'max_c': max_c
                })

    # Sort rectangles by reading order (row then column)
    all_rects.sort(key=lambda r: (r['min_r'], r['min_c']))

    # Map sequence to rectangles
    color_count = {}
    rect_sequence = []

    for color in key_colors:
        count = color_count.get(color, 0)
        # Find the (count+1)th rectangle with this color
        matching_rects = [r for r in all_rects if r['color'] == color]
        if count < len(matching_rects):
            rect_sequence.append(matching_rects[count])
            color_count[color] = count + 1

    # Process consecutive pairs
    for i in range(len(rect_sequence) - 1):
        rect1 = rect_sequence[i]
        rect2 = rect_sequence[i + 1]

        # Check alignment
        row_aligned = (rect1['min_r'] <= rect2['max_r'] and rect2['min_r'] <= rect1['max_r'])
        col_aligned = (rect1['min_c'] <= rect2['max_c'] and rect2['min_c'] <= rect1['max_c'])

        if row_aligned and not col_aligned:
            # Horizontal connection - extend color1 toward color2
            overlap_min_r = max(rect1['min_r'], rect2['min_r'])
            overlap_max_r = min(rect1['max_r'], rect2['max_r'])

            if rect1['max_c'] < rect2['min_c']:
                # rect1 is to the left
                for r in range(overlap_min_r, overlap_max_r + 1):
                    for c in range(rect1['max_c'] + 1, rect2['min_c']):
                        if result[r][c] == background:
                            result[r][c] = rect1['color']
            else:
                # rect1 is to the right
                for r in range(overlap_min_r, overlap_max_r + 1):
                    for c in range(rect2['max_c'] + 1, rect1['min_c']):
                        if result[r][c] == background:
                            result[r][c] = rect1['color']

        elif col_aligned and not row_aligned:
            # Vertical connection - place color1 in gap
            overlap_min_c = max(rect1['min_c'], rect2['min_c'])
            overlap_max_c = min(rect1['max_c'], rect2['max_c'])

            if rect1['max_r'] < rect2['min_r']:
                # rect1 is above
                for r in range(rect1['max_r'] + 1, rect2['min_r']):
                    for c in range(overlap_min_c, overlap_max_c + 1):
                        if result[r][c] == background:
                            result[r][c] = rect1['color']
            else:
                # rect1 is below
                for r in range(rect2['max_r'] + 1, rect1['min_r']):
                    for c in range(overlap_min_c, overlap_max_c + 1):
                        if result[r][c] == background:
                            result[r][c] = rect1['color']

        elif not row_aligned and not col_aligned:
            # Diagonal - extend both toward middle
            mid_r = (rect1['max_r'] + rect2['min_r']) // 2 if rect1['max_r'] < rect2['min_r'] else (rect2['max_r'] + rect1['min_r']) // 2

            # Extend rect1 vertically
            for c in range(rect1['min_c'], rect1['max_c'] + 1):
                if rect1['max_r'] < rect2['min_r']:
                    for r in range(rect1['max_r'] + 1, mid_r + 1):
                        if result[r][c] == background:
                            result[r][c] = rect1['color']
                else:
                    for r in range(mid_r, rect1['min_r']):
                        if result[r][c] == background:
                            result[r][c] = rect1['color']

            # Extend rect2 vertically
            for c in range(rect2['min_c'], rect2['max_c'] + 1):
                if rect2['max_r'] < rect1['min_r']:
                    for r in range(rect2['max_r'] + 1, mid_r + 1):
                        if result[r][c] == background:
                            result[r][c] = rect2['color']
                else:
                    for r in range(mid_r, rect2['min_r']):
                        if result[r][c] == background:
                            result[r][c] = rect2['color']

    return result
