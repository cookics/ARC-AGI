def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has regions with different dominant colors
    2. Isolated cells (>60% of neighbors differ) trigger vertical line segments
    3. Line segments include isolated cell + vertically consecutive cells with same value
    4. Line colors are swapped between regions

    Procedure:
    1. Find all isolated cells
    2. For each isolated cell, find vertically consecutive cells with same value
    3. Change these cells based on region (top uses bottom's dominant color, vice versa)
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find isolated cells (anomalous cells different from surroundings)
    def is_isolated(r, c):
        val = grid[r][c]
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(grid[nr][nc])

        if len(neighbors) < 5:  # Relaxed edge constraint
            return False

        different = sum(1 for n in neighbors if n != val)
        # Lower threshold to catch more isolated cells
        return different >= len(neighbors) * 0.5

    # Find all isolated cells
    isolated_cells = set()
    for r in range(rows):
        for c in range(cols):
            if is_isolated(r, c):
                isolated_cells.add((r, c))

    # For each isolated cell, find vertical segment
    processed = set()
    segments = []

    for r, c in isolated_cells:
        if (r, c) in processed:
            continue

        val = grid[r][c]
        segment = [(r, c)]
        processed.add((r, c))

        # Expand upward
        rr = r - 1
        while rr >= 0 and grid[rr][c] == val:
            segment.append((rr, c))
            processed.add((rr, c))
            rr -= 1

        # Expand downward
        rr = r + 1
        while rr < rows and grid[rr][c] == val:
            segment.append((rr, c))
            processed.add((rr, c))
            rr += 1

        segments.append((segment, val, c))

    # Identify regions and dominant colors
    mid_row = rows // 2

    from collections import Counter

    top_colors = Counter()
    bottom_colors = Counter()

    for r in range(mid_row):
        for c in range(cols):
            top_colors[grid[r][c]] += 1

    for r in range(mid_row, rows):
        for c in range(cols):
            bottom_colors[grid[r][c]] += 1

    top_doms = [color for color, count in top_colors.most_common(5)]
    bottom_doms = [color for color, count in bottom_colors.most_common(5)]

    # Change each segment
    for segment, val, col in segments:
        for r, c in segment:
            # Determine replacement color based on region
            if r < mid_row:
                # Top region - use non-dominant color from bottom (prefer 2nd most common)
                candidates = [color for color in bottom_doms if color != val]
                if candidates:
                    # Use 2nd candidate if available (skip most dominant background)
                    result[r][c] = candidates[min(1, len(candidates)-1)]
            else:
                # Bottom region - use non-dominant color from top
                candidates = [color for color in top_doms if color != val]
                if candidates:
                    result[r][c] = candidates[min(1, len(candidates)-1)]

    return result
