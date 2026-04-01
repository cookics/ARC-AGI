def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has TWO overlapping frame structures (patterns)
    2. Extract and merge these patterns by overlaying them
    3. Output has horizontal symmetry and some row duplication

    Procedure:
    1. Find connected components (objects) using BFS
    2. Identify two main objects by size/position
    3. Extract bounding box and merge overlapping regions
    4. Apply horizontal symmetry to each row
    5. Duplicate rows containing special colors
    """
    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])

    # Find background
    flat = [grid[i][j] for i in range(rows) for j in range(cols)]
    background = Counter(flat).most_common()[0][0]

    # Find connected components using BFS
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(start_r, start_c):
        comp = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            comp.append((r, c, grid[r][c]))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    not visited[nr][nc] and grid[nr][nc] != background):
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return comp

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid[i][j] != background:
                comp = bfs(i, j)
                if comp:
                    components.append(comp)

    if not components:
        return grid

    # Find the largest component (usually contains both patterns)
    largest_comp = max(components, key=len)

    # Get bounding box
    min_r = min(r for r, c, v in largest_comp)
    max_r = max(r for r, c, v in largest_comp)
    min_c = min(c for r, c, v in largest_comp)
    max_c = max(c for r, c, v in largest_comp)

    # Extract region and create output with horizontal symmetry
    result = []

    for row_idx in range(min_r, max_r + 1):
        # Extract this row
        row_data = [grid[row_idx][c] for c in range(min_c, max_c + 1)]

        # Make row horizontally symmetric
        width = len(row_data)
        mid = width // 2

        symmetric_row = row_data[:]
        for i in range(mid):
            left_val = symmetric_row[i]
            right_val = symmetric_row[width - 1 - i]

            # Merge: prefer non-background values
            if left_val != background and right_val != background:
                # Both have values, keep them as is (already part of pattern)
                pass
            elif left_val != background:
                symmetric_row[width - 1 - i] = left_val
            elif right_val != background:
                symmetric_row[i] = right_val

        result.append(symmetric_row)

    # Apply row duplication for rows with rare colors
    color_freq = Counter(v for row in result for v in row if v != background)
    rare_colors = set(c for c, count in color_freq.items() if count < len(result) * 2)

    final_result = []
    for i, row in enumerate(result):
        final_result.append(row[:])

        # Duplicate rows with rare colors
        if any(val in rare_colors for val in row):
            if i == len(result) - 1 or result[i] != result[i + 1]:
                # Duplicate this row
                final_result.append(row[:])

    return final_result
