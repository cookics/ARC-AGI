def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a background color (most frequent) and scattered non-background patterns
    2. Each pattern consists of cells with one dominant value, plus exactly one "marker" cell with a different value
    3. Patterns get swapped: pattern cells become marker value, marker cell becomes pattern value
    4. Corner cells (non-background) act as anchors - patterns with matching marker get copied to corners
    5. The copied pattern is placed at the corner with all non-background cells using the marker value

    Procedure:
    1. Find background value (most frequent)
    2. Identify corner markers (non-background values at four corners)
    3. Find all connected component patterns (excluding corners themselves initially)
    4. For each pattern, identify the marker value (rarest non-background value in the pattern)
    5. Swap pattern values with marker values
    6. For each corner marker, find closest pattern with that marker and copy it to the corner
    """

    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find background value
    all_values = [val for row in grid for val in row]
    background = Counter(all_values).most_common(1)[0][0]

    # Get corner markers
    corners = {}
    corner_positions = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    for r, c in corner_positions:
        if grid[r][c] != background:
            corners[(r, c)] = grid[r][c]

    # Find connected components (excluding single-cell patterns at corners)
    visited = [[False] * cols for _ in range(rows)]
    patterns = []

    def bfs(start_r, start_c):
        """Find connected component of non-background cells using 8-connectivity"""
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            # Use 8-connectivity (including diagonals)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not visited[nr][nc]
                        and grid[nr][nc] != background
                    ):
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return component

    # Find all patterns
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != background and not visited[i][j]:
                component = bfs(i, j)
                if len(component) > 1:  # Ignore isolated single cells
                    patterns.append(component)
                elif len(component) == 1 and (i, j) not in corners:
                    # Mark as visited but don't add as pattern
                    pass

    # Process each pattern: identify marker and swap
    pattern_info = []  # (pattern_cells, pattern_value, marker_value, marker_pos, bbox)

    for pattern in patterns:
        # Count values in this pattern
        value_counts = Counter([grid[r][c] for r, c in pattern])

        if len(value_counts) < 2:
            continue  # No marker, skip

        # Pattern value is most common, marker is least common (and different)
        sorted_values = value_counts.most_common()
        pattern_value = sorted_values[0][0]
        marker_value = sorted_values[-1][0]

        # Find marker position
        marker_pos = None
        for r, c in pattern:
            if grid[r][c] == marker_value and grid[r][c] != pattern_value:
                marker_pos = (r, c)
                break

        if marker_pos is None:
            continue

        # Calculate bounding box
        min_r = min(r for r, c in pattern)
        max_r = max(r for r, c in pattern)
        min_c = min(c for r, c in pattern)
        max_c = max(c for r, c in pattern)
        bbox = (min_r, max_r, min_c, max_c)

        pattern_info.append((pattern, pattern_value, marker_value, marker_pos, bbox))

        # Perform swap in result
        for r, c in pattern:
            if (r, c) == marker_pos:
                result[r][c] = pattern_value
            else:
                result[r][c] = marker_value

    # Copy patterns to corners (copy ALL patterns with matching marker, not just closest)
    for (corner_r, corner_c), corner_marker in corners.items():
        # Find patterns with this marker value
        matching_patterns = [
            (pat, pval, mval, mpos, bbox)
            for pat, pval, mval, mpos, bbox in pattern_info
            if mval == corner_marker
        ]

        if not matching_patterns:
            continue

        # Copy ALL matching patterns to the corner (overlay them)
        for pat, pval, mval, mpos, bbox in matching_patterns:
            min_r, max_r, min_c, max_c = bbox

            # Create relative pattern
            pattern_height = max_r - min_r + 1
            pattern_width = max_c - min_c + 1

            relative_pattern = set()
            for r, c in pat:
                rel_r = r - min_r
                rel_c = c - min_c
                relative_pattern.add((rel_r, rel_c))

            # Place pattern at corner
            if corner_r == 0 and corner_c == 0:  # Top-left
                offset_r, offset_c = 0, 0
            elif corner_r == 0 and corner_c == cols - 1:  # Top-right
                offset_r, offset_c = 0, cols - pattern_width
            elif corner_r == rows - 1 and corner_c == 0:  # Bottom-left
                offset_r, offset_c = rows - pattern_height, 0
            else:  # Bottom-right
                offset_r, offset_c = rows - pattern_height, cols - pattern_width

            # Draw pattern at corner location
            for rel_r, rel_c in relative_pattern:
                abs_r = offset_r + rel_r
                abs_c = offset_c + rel_c
                if 0 <= abs_r < rows and 0 <= abs_c < cols:
                    result[abs_r][abs_c] = corner_marker

    return result
