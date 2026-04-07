def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a rectangular template filled with 1s (with some 4s inside)
    2. Input has scattered 4s throughout the grid
    3. Template is replicated at locations where clusters of 4s exist
    4. When placing, fill with 1s but PRESERVE existing 4s

    Procedure:
    1. Find template and extract its pattern
    2. Find scattered 4s using BFS clustering with 4-connectivity
    3. For each cluster, place template with bbox-based alignment
    4. Apply template filling with 1s, preserving 4s
    """
    import copy
    from collections import deque

    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find template (bounding box of all 1s)
    ones_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not ones_cells:
        return result

    min_r = min(r for r, c in ones_cells)
    max_r = max(r for r, c in ones_cells)
    min_c = min(c for r, c in ones_cells)
    max_c = max(c for r, c in ones_cells)

    template_height = max_r - min_r + 1
    template_width = max_c - min_c + 1

    # Find scattered 4s (outside template region)
    scattered_4s = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                if not (min_r <= r <= max_r and min_c <= c <= max_c):
                    scattered_4s.add((r, c))

    # Cluster using BFS with extended 4-connectivity (distance 1-2)
    visited = set()
    clusters = []

    for start in scattered_4s:
        if start in visited:
            continue

        cluster = []
        queue = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            cluster.append((r, c))

            # Check neighbors with extended 4-connectivity (distance 1-2)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0),
                          (0, 2), (0, -2), (2, 0), (-2, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in scattered_4s and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        # Only keep clusters with 2+ members
        if len(cluster) >= 2:
            clusters.append(cluster)

    # Place templates with bbox expansion
    for cluster in clusters:
        min_cr = min(r for r, c in cluster)
        max_cr = max(r for r, c in cluster)
        min_cc = min(c for r, c in cluster)
        max_cc = max(c for r, c in cluster)

        # Check if cluster fits in template
        cluster_height = max_cr - min_cr + 1
        cluster_width = max_cc - min_cc + 1

        if cluster_height > template_height or cluster_width > template_width:
            continue

        # Calculate placement with symmetric expansion
        vertical_padding = template_height - cluster_height
        horizontal_padding = template_width - cluster_width

        start_r = min_cr - vertical_padding // 2
        start_c = min_cc - horizontal_padding // 2

        # Ensure within bounds
        start_r = max(0, min(start_r, rows - template_height))
        start_c = max(0, min(start_c, cols - template_width))

        # Skip if overlaps original template
        if (start_r <= max_r and start_r + template_height > min_r and
            start_c <= max_c and start_c + template_width > min_c):
            continue

        # Apply template: fill with 1s, preserve 4s
        for dr in range(template_height):
            for dc in range(template_width):
                r = start_r + dr
                c = start_c + dc
                if 0 <= r < rows and 0 <= c < cols:
                    if result[r][c] != 4:
                        result[r][c] = 1

    return result
