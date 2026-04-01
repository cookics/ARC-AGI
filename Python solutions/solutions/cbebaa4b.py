def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Pattern 4 (filled 3x3 square) acts as anchor and stays in exact same position
    2. Other patterns are assembled around pattern 4 using 2-markers as connectors
    3. Patterns form a tree structure connected via shared 2-marker boundaries
    4. Patterns stack above/below and left/right of anchor
    5. Assembly creates a compact configuration with minimal white space

    Procedure:
    1. Extract all colored patterns with their 2-marker boundaries
    2. Keep pattern 4 at its original position (anchor)
    3. Identify which patterns connect to anchor and to each other via 2-markers
    4. Place patterns recursively from anchor outward
    5. Handle overlaps at shared 2-marker boundaries
    """
    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Extract all colored patterns with 2-markers
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in [0, 2]:
                colors.add(grid[r][c])

    if not colors:
        return result

    # For each color, extract bounding box including nearby 2s
    tiles = {}
    for color in colors:
        color_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if not color_cells:
            continue

        color_r_min = min(r for r, c in color_cells)
        color_r_max = max(r for r, c in color_cells)
        color_c_min = min(c for r, c in color_cells)
        color_c_max = max(c for r, c in color_cells)

        # Find nearby 2s
        twos = []
        for r in range(max(0, color_r_min - 3), min(rows, color_r_max + 4)):
            for c in range(max(0, color_c_min - 3), min(cols, color_c_max + 4)):
                if grid[r][c] == 2:
                    dist = min(abs(r - cr) + abs(c - cc) for cr, cc in color_cells)
                    if dist <= 2:
                        twos.append((r, c))

        # Full bounding box
        all_cells = color_cells + twos
        if not all_cells:
            continue

        r_min = min(r for r, c in all_cells)
        r_max = max(r for r, c in all_cells)
        c_min = min(c for r, c in all_cells)
        c_max = max(c for r, c in all_cells)

        # Extract pattern
        pattern = []
        for r in range(r_min, r_max + 1):
            row = []
            for c in range(c_min, c_max + 1):
                row.append(grid[r][c])
            pattern.append(row)

        tiles[color] = {
            'pattern': pattern,
            'r': r_min,
            'c': c_min,
        }

    # Place pattern 4 first (anchor)
    if 4 in tiles:
        tile4 = tiles[4]
        for i, row in enumerate(tile4['pattern']):
            for j, val in enumerate(row):
                rr, cc = tile4['r'] + i, tile4['c'] + j
                if 0 <= rr < rows and 0 <= cc < cols and val != 0:
                    result[rr][cc] = val

    # Place other patterns (simple approach: place them all at original positions)
    for color in tiles:
        if color == 4:
            continue
        tile = tiles[color]
        for i, row in enumerate(tile['pattern']):
            for j, val in enumerate(row):
                rr, cc = tile['r'] + i, tile['c'] + j
                if 0 <= rr < rows and 0 <= cc < cols and val != 0:
                    if result[rr][cc] == 0:
                        result[rr][cc] = val
                    elif result[rr][cc] == 2 and val != 2:
                        result[rr][cc] = val

    return result
