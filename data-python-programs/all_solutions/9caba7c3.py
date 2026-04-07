def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 0, 5 (background), and 2 (markers)
    2. Find connected components of 2s using 8-connected adjacency
    3. For each component, expand to minimal odd-sized bounding box
    4. Within the box: one 5 becomes 4 (center or most adjacent to 2s), others become 7

    Procedure:
    1. Find all 2s and group them into 8-connected components
    2. For each component:
       a. Compute bounding box
       b. Expand to odd-sized (3x3 minimum)
       c. Mark non-2 cells within box: center→4, others→7
    """

    result = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    # Find all 2s
    twos = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.add((r, c))

    # Union-find for 8-connected components
    parent = {pos: pos for pos in twos}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Connect 8-adjacent 2s
    for r, c in twos:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in twos:
                    union((r, c), (nr, nc))

    # Group components
    component_map = {}
    for pos in twos:
        root = find(pos)
        if root not in component_map:
            component_map[root] = []
        component_map[root].append(pos)

    components = list(component_map.values())

    # Process each component
    for component in components:
        # Find bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        # Expand to odd-sized box (minimum 3x3)
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # Make dimensions odd (at least 3)
        target_height = max(3, height if height % 2 == 1 else height + 1)
        target_width = max(3, width if width % 2 == 1 else width + 1)

        expand_h = target_height - height
        expand_w = target_width - width

        # Try all valid expansions and choose the best
        candidates = []
        for exp_up in range(min(expand_h + 1, min_r + 1)):
            exp_down = expand_h - exp_up
            if max_r + exp_down >= rows:
                continue

            for exp_left in range(min(expand_w + 1, min_c + 1)):
                exp_right = expand_w - exp_left
                if max_c + exp_right >= cols:
                    continue

                test_min_r = min_r - exp_up
                test_max_r = max_r + exp_down
                test_min_c = min_c - exp_left
                test_max_c = max_c + exp_right

                # Compute center of expanded box
                center_r = (test_min_r + test_max_r) // 2
                center_c = (test_min_c + test_max_c) // 2

                # Prefer center not being 2, then balanced, then down+right
                center_is_non_2 = (grid[center_r][center_c] != 2)
                asymmetry = abs(exp_up - exp_down) + abs(exp_left - exp_right)

                candidates.append((
                    center_is_non_2,
                    -asymmetry,  # prefer balanced
                    -(exp_up + exp_left),  # prefer down+right (lower exp_up+exp_left)
                    test_min_r, test_max_r, test_min_c, test_max_c
                ))

        if candidates:
            candidates.sort(reverse=True)
            _, _, _, new_min_r, new_max_r, new_min_c, new_max_c = candidates[0]
        else:
            # Fallback
            new_min_r = min_r
            new_max_r = max_r
            new_min_c = min_c
            new_max_c = max_c

        # Calculate center of the box
        center_r = (new_min_r + new_max_r) // 2
        center_c = (new_min_c + new_max_c) // 2

        # Mark all non-2 cells in the box
        for r in range(new_min_r, new_max_r + 1):
            for c in range(new_min_c, new_max_c + 1):
                if grid[r][c] != 2:
                    if r == center_r and c == center_c:
                        result[r][c] = 4
                    else:
                        result[r][c] = 7

    return result
