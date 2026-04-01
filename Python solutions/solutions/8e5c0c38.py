def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a dominant background color
    2. There are patterns/shapes of other colors on the background
    3. Remove cells on bbox edge with <= 2 neighbors (but some exceptions)

    Procedure:
    1. Identify the background color (most frequent)
    2. For each color, find cells on bbox boundary with weak connectivity
    """

    from collections import Counter

    # Deep copy
    result = [row[:] for row in grid]

    # Find background color (most frequent)
    flat = [cell for row in grid for cell in row]
    background = Counter(flat).most_common(1)[0][0]

    # Find all cells for each color
    color_cells = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            color = grid[i][j]
            if color == background:
                continue
            if color not in color_cells:
                color_cells[color] = []
            color_cells[color].append((i, j))

    # Process each color
    to_remove = []

    for color, cells in color_cells.items():
        if not cells:
            continue

        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        cells_set = set(cells)

        for r, c in cells:
            # Check if on boundary of bounding box
            on_boundary = (r == min_r or r == max_r or c == min_c or c == max_c)

            if not on_boundary:
                continue

            # Count 4-neighbors with same color
            neighbor_count = sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                               if (r+dr, c+dc) in cells_set)

            # Remove if has <= 2 neighbors AND on extreme edge
            on_extreme_edge = ((r == min_r or r == max_r) and (c == min_c or c == max_c))

            if neighbor_count <= 1 or (neighbor_count == 2 and on_extreme_edge):
                to_remove.append((r, c))

    for i, j in to_remove:
        result[i][j] = background

    return result
