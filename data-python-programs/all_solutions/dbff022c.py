def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has hollow rectangles with uniform border colors
    2. A scattered region contains key-value pairs (small rectangular area)
    3. For each hollow rectangle with border color K, if pair (K,V) exists, fill interior 0s with V
    4. Pairing method depends on scattered region shape:
       - More rows than columns: pair by rows (col 0 = key, col 1 = value)
       - More columns than rows: pair by columns (row 0 = key, row 1 = value)

    Procedure:
    1. Find hollow rectangles (uniform border, interior may have 0s)
    2. Find scattered region (small dense area with varied values)
    3. Extract key-value pairs from scattered region
    4. Fill rectangles according to the mapping
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find connected components using flood fill
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def flood_fill(start_r, start_c, color):
        cells = []
        stack = [(start_r, start_c)]
        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != color:
                continue
            visited[r][c] = True
            cells.append((r, c))
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))
        return cells

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                cells = flood_fill(r, c, grid[r][c])
                components.append({
                    'color': grid[r][c],
                    'cells': set(cells)
                })

    # Find hollow rectangles from components
    hollow_rects = []

    for comp in components:
        cells = comp['cells']
        if len(cells) < 4:
            continue

        color = comp['color']

        # Get bounding box
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Check if this forms a rectangular border
        # Collect interior positions that should be filled
        interior = []

        for r in range(min_r, max_r + 1):
            # Find leftmost and rightmost cells of this color in this row (within bounding box)
            row_cells = []
            for c in range(min_c, max_c + 1):
                if grid[r][c] == color:
                    row_cells.append(c)

            if len(row_cells) >= 2:
                # Interior is between leftmost and rightmost
                for c in range(row_cells[0] + 1, row_cells[-1]):
                    if grid[r][c] == 0:
                        interior.append((r, c))


        if interior:
            hollow_rects.append({
                'color': color,
                'cells': cells,
                'interior': interior,
                'bbox': (min_r, max_r, min_c, max_c)
            })

    # Find scattered region (cells not part of any hollow rectangle)
    rect_cells = set()
    for rect in hollow_rects:
        rect_cells.update(rect['cells'])

    scattered_cells = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in rect_cells:
                scattered_cells.append((r, c))

    # Find bounding box of scattered region
    if scattered_cells:
        s_min_r = min(r for r, c in scattered_cells)
        s_max_r = max(r for r, c in scattered_cells)
        s_min_c = min(c for r, c in scattered_cells)
        s_max_c = max(c for r, c in scattered_cells)

        s_height = s_max_r - s_min_r + 1
        s_width = s_max_c - s_min_c + 1

        # Extract key-value pairs
        mapping = {}

        if s_height > s_width:
            # Pair by rows: each row is (key, value)
            for r in range(s_min_r, s_max_r + 1):
                row_vals = []
                for c in range(s_min_c, s_max_c + 1):
                    if grid[r][c] != 0:
                        row_vals.append(grid[r][c])
                if len(row_vals) >= 2:
                    mapping[row_vals[0]] = row_vals[1]
        else:
            # Pair by columns: each column is (key, value)
            for c in range(s_min_c, s_max_c + 1):
                col_vals = []
                for r in range(s_min_r, s_max_r + 1):
                    if grid[r][c] != 0:
                        col_vals.append(grid[r][c])
                if len(col_vals) >= 2:
                    mapping[col_vals[0]] = col_vals[1]

        # Fill rectangles according to mapping
        for rect in hollow_rects:
            color = rect['color']
            if color in mapping:
                fill_value = mapping[color]
                for r, c in rect['interior']:
                    result[r][c] = fill_value

    return result
