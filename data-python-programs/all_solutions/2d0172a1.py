def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with two colors where the minority color forms a noisy pattern
    2. The pattern contains topological features: loops, holes, and nested structures
    3. Output is a clean, compact representation showing:
       - The outer boundary of the pattern as a rectangle
       - Internal holes/loops as clean rectangular regions
       - The topological structure preserved
    4. This is essentially extracting the "skeleton" or canonical form of the pattern

    Procedure:
    1. Identify the two colors (background vs pattern)
    2. Find bounding box of pattern
    3. Detect holes in the pattern using flood fill from outside
    4. Create clean output representing the pattern structure with rectangles
    """
    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])

    # Identify colors by frequency
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)
    colors = sorted(counts.keys(), key=lambda x: counts[x])
    pattern_color = colors[0]  # Minority
    background_color = colors[1]  # Majority

    # Find bounding box of pattern
    pattern_cells = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == pattern_color:
                pattern_cells.add((r, c))

    if not pattern_cells:
        return [[background_color]]

    min_r = min(r for r, c in pattern_cells)
    max_r = max(r for r, c in pattern_cells)
    min_c = min(c for r, c in pattern_cells)
    max_c = max(c for r, c in pattern_cells)

    # Create local coordinate system
    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # Build binary mask (1 = pattern, 0 = background)
    mask = [[0] * w for _ in range(h)]
    for r, c in pattern_cells:
        mask[r - min_r][c - min_c] = 1

    # Detect holes using flood fill from outside
    # Add padding for flood fill
    padded_h = h + 2
    padded_w = w + 2
    padded = [[0] * padded_w for _ in range(padded_h)]
    for r in range(h):
        for c in range(w):
            padded[r + 1][c + 1] = mask[r][c]

    # Flood fill from (0,0) to mark all background cells reachable from outside
    outside = [[False] * padded_w for _ in range(padded_h)]
    queue = deque([(0, 0)])
    outside[0][0] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < padded_h and 0 <= nc < padded_w:
                if not outside[nr][nc] and padded[nr][nc] == 0:
                    outside[nr][nc] = True
                    queue.append((nr, nc))

    # Find holes (background cells not reachable from outside)
    holes = []
    for r in range(1, padded_h - 1):
        for c in range(1, padded_w - 1):
            if padded[r][c] == 0 and not outside[r][c]:
                holes.append((r - 1, c - 1))  # Convert back to local coords

    # Find connected components of holes
    hole_set = set(holes)
    hole_visited = set()
    hole_components = []

    def bfs_hole(start_r, start_c):
        component = []
        q = deque([(start_r, start_c)])
        hole_visited.add((start_r, start_c))

        while q:
            r, c = q.popleft()
            component.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in hole_set and (nr, nc) not in hole_visited:
                    hole_visited.add((nr, nc))
                    q.append((nr, nc))

        return component

    for hr, hc in holes:
        if (hr, hc) not in hole_visited:
            comp = bfs_hole(hr, hc)
            hole_components.append(comp)

    # Build output representing the structure
    # Simple approach: create output showing boundary and holes
    result = [[background_color] * w for _ in range(h)]

    # Mark pattern cells
    for r in range(h):
        for c in range(w):
            if mask[r][c] == 1:
                result[r][c] = pattern_color

    # Mark holes with pattern color (representing the loop structure)
    for hole_comp in hole_components:
        if len(hole_comp) >= 1:  # At least one cell
            # Get bounding box of this hole
            hole_r_min = min(r for r, c in hole_comp)
            hole_r_max = max(r for r, c in hole_comp)
            hole_c_min = min(c for r, c in hole_comp)
            hole_c_max = max(c for r, c in hole_comp)

            # Mark center of hole with pattern color
            hole_r_center = (hole_r_min + hole_r_max) // 2
            hole_c_center = (hole_c_min + hole_c_max) // 2
            if 0 <= hole_r_center < h and 0 <= hole_c_center < w:
                result[hole_r_center][hole_c_center] = pattern_color

    return result
