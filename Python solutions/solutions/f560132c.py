def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 4 colored regions: one container region with a 2x2 pattern inside, and 3 other regions
    2. The 2x2 pattern contains 4 distinct non-zero values
    3. Output size is ceil(sqrt(sum of cells in 3 non-container regions)) + 1
    4. Output is a square grid where each cell is assigned to one of the 4 pattern values
    5. Assignment is based on Euclidean distance to corners, weighted by region sizes
    6. Pattern values map to regions based on spatial position (quadrants)

    Procedure:
    1. Find 2x2 block with 4 distinct non-zero values (the pattern)
    2. Identify container color (surrounds the pattern)
    3. Find 3 other colored regions and count their cells
    4. Map pattern positions to regions by quadrant
    5. Calculate output size and quotas
    6. Fill output using power diagram (distance^2 / quota)
    """
    import math
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Find 2x2 pattern
    pattern = None
    pattern_pos = None
    for r in range(rows - 1):
        for c in range(cols - 1):
            vals = [grid[r][c], grid[r][c+1], grid[r+1][c], grid[r+1][c+1]]
            if len(set(vals)) == 4 and all(v != 0 for v in vals):
                pattern = [[grid[r][c], grid[r][c+1]], [grid[r+1][c], grid[r+1][c+1]]]
                pattern_pos = (r, c)
                break
        if pattern:
            break

    # Find container color (most common color adjacent to pattern)
    pattern_values = set([pattern[0][0], pattern[0][1], pattern[1][0], pattern[1][1]])
    pr, pc = pattern_pos
    adjacent_colors = []
    for dr, dc in [(0, -1), (-1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
        nr, nc = pr + dr, pc + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] not in pattern_values and grid[nr][nc] != 0:
            adjacent_colors.append(grid[nr][nc])
    container_color = Counter(adjacent_colors).most_common(1)[0][0] if adjacent_colors else None

    # Count cells by color
    color_counts = Counter()
    for row in grid:
        for cell in row:
            if cell != 0 and cell not in pattern_values:
                color_counts[cell] += 1

    # Calculate centroids for each color
    color_positions = {}
    color_cells = {}
    for color in color_counts:
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        color_cells[color] = positions
        if positions:
            avg_r = sum(r for r, c in positions) / len(positions)
            avg_c = sum(c for r, c in positions) / len(positions)
            color_positions[color] = (avg_r, avg_c)

    # Find global center
    all_colored = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]
    center_r = sum(r for r, c in all_colored) / len(all_colored)
    center_c = sum(c for r, c in all_colored) / len(all_colored)

    # Map colors to quadrants
    quadrant_colors = {'TL': None, 'TR': None, 'BL': None, 'BR': None}
    for color in color_counts:
        if color == container_color or color in pattern_values:
            continue
        r, c = color_positions[color]
        is_top = r < center_r
        is_left = c < center_c
        quadrant = ('T' if is_top else 'B') + ('L' if is_left else 'R')
        if quadrant_colors[quadrant] is None:
            quadrant_colors[quadrant] = color

    # Assign container to remaining quadrant or TL
    if container_color:
        r, c = color_positions[container_color]
        is_top = r < center_r
        is_left = c < center_c
        quadrant = ('T' if is_top else 'B') + ('L' if is_left else 'R')
        quadrant_colors[quadrant] = container_color

    # Map pattern values to colors
    val_tl, val_tr, val_bl, val_br = pattern[0][0], pattern[0][1], pattern[1][0], pattern[1][1]

    # Build quota mapping
    quotas = {}
    for quadrant, val in [('TL', val_tl), ('TR', val_tr), ('BL', val_bl), ('BR', val_br)]:
        color = quadrant_colors[quadrant]
        if color:
            quotas[val] = color_counts[color]

    # Calculate output size (excluding container)
    non_container_cells = sum(count for color, count in color_counts.items() if color != container_color)
    n = math.ceil(math.sqrt(non_container_cells)) + 1
    total_output_cells = n * n

    # Container gets extra cells
    for val in quotas:
        if container_color and quotas[val] == color_counts.get(container_color, 0):
            quotas[val] = total_output_cells - sum(q for v, q in quotas.items() if v != val)
            break

    # Fill output using power diagram with weight = quota
    result = [[0] * n for _ in range(n)]
    corners = {
        val_tl: (0, 0),
        val_tr: (0, n-1),
        val_bl: (n-1, 0),
        val_br: (n-1, n-1)
    }

    for r in range(n):
        for c in range(n):
            min_power_dist = float('inf')
            best_val = val_tl

            for val, (cr, cc) in corners.items():
                if val in quotas and quotas[val] > 0:
                    # Power distance = distance^2 - weight
                    dist_sq = (r - cr) ** 2 + (c - cc) ** 2
                    weight = quotas[val] * 1.2  # scale factor
                    power_dist = dist_sq - weight

                    if power_dist < min_power_dist:
                        min_power_dist = power_dist
                        best_val = val

            result[r][c] = best_val

    return result
