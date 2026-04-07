def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains exactly one 2×2 block of 2s (the anchor)
    2. Input contains multiple 2×2 blocks of 8s
    3. Output preserves all blocks and adds 7s to connect them
    4. The 7s form a spanning tree rooted at the 2s block
    5. Blocks are connected horizontally/vertically if they share row/column
    6. Use DFS/BFS to build the spanning tree

    Procedure:
    1. Find all 2×2 blocks (8s and 2s)
    2. Build adjacency graph (blocks on same row/col with no blocks between)
    3. DFS from the 2s block to create spanning tree
    4. For each edge in tree, fill gap with 7s
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find all 2×2 blocks
    blocks = []
    anchor = None

    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check if this is a 2×2 block
            val = grid[r][c]
            if val != 0 and all(grid[r + dr][c + dc] == val for dr in [0, 1] for dc in [0, 1]):
                if val == 2:
                    anchor = (r, c)
                elif val == 8:
                    blocks.append((r, c))

    if anchor is None:
        return grid

    blocks.append(anchor)

    # Build adjacency list (blocks that can be directly connected)
    adj = {block: [] for block in blocks}

    for i, b1 in enumerate(blocks):
        for j, b2 in enumerate(blocks):
            if i >= j:
                continue

            r1, c1 = b1
            r2, c2 = b2

            # Check if on same row or column
            if r1 == r2:
                # Same row - check no blocks between them
                min_c, max_c = min(c1, c2), max(c1, c2)
                between = any(r == r1 and min_c < c < max_c for r, c in blocks if (r, c) != b1 and (r, c) != b2)
                if not between:
                    adj[b1].append(b2)
                    adj[b2].append(b1)
            elif c1 == c2:
                # Same column - check no blocks between them
                min_r, max_r = min(r1, r2), max(r1, r2)
                between = any(c == c1 and min_r < r < max_r for r, c in blocks if (r, c) != b1 and (r, c) != b2)
                if not between:
                    adj[b1].append(b2)
                    adj[b2].append(b1)

    # DFS from anchor to build spanning tree
    # Sort neighbors by distance to prefer closer connections
    visited = set()
    edges = []

    def dfs(node):
        visited.add(node)
        r1, c1 = node

        # Sort neighbors with special handling for the anchor
        if node == anchor:
            # From anchor (2s block), prefer: right, down, left, up
            neighbors = sorted(adj[node], key=lambda n: (
                not (n[0] == r1 and n[1] > c1),  # Right first (same row, greater col)
                not (n[0] > r1 and n[1] == c1),  # Then down (same col, greater row)
                not (n[0] == r1 and n[1] < c1),  # Then left
                not (n[0] < r1 and n[1] == c1),  # Then up
                abs(n[0] - r1) + abs(n[1] - c1),  # Then by distance
            ))
            # Only connect to the first (closest in preferred direction)
            for i, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    edges.append((node, neighbor))
                    dfs(neighbor)
                    if i == 0:  # Only explore one neighbor from anchor
                        break
        else:
            # From other blocks, sort by distance
            neighbors = sorted(adj[node], key=lambda n: (
                abs(n[0] - r1) + abs(n[1] - c1),  # Distance
                n[0] != r1,  # Prefer horizontal over vertical
                n[1],  # Then by column
                n[0]   # Then by row
            ))

            for neighbor in neighbors:
                if neighbor not in visited:
                    edges.append((node, neighbor))
                    dfs(neighbor)

    dfs(anchor)

    # Create result grid
    result = [row[:] for row in grid]

    # Fill gaps with 7s
    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            # Horizontal connection
            # Block 1: cols c1, c1+1
            # Block 2: cols c2, c2+1
            # Fill gap between them
            min_c, max_c = min(c1, c2), max(c1, c2)
            for c in range(min_c + 2, max_c):
                result[r1][c] = 7
                result[r1 + 1][c] = 7
        else:
            # Vertical connection
            # Block 1: rows r1, r1+1
            # Block 2: rows r2, r2+1
            # Fill gap between them
            min_r, max_r = min(r1, r2), max(r1, r2)
            for r in range(min_r + 2, max_r):
                result[r][c1] = 7
                result[r][c1 + 1] = 7

    return result
