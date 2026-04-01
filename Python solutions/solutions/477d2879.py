def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has markers (> 1), walls (1), and empty cells (0)
    2. One marker may be adjacent to 1s - this colors all 1s (if clear winner)
    3. Other markers color regions of 0s through simultaneous BFS
    4. 1s colored by most frequent adjacent color if no structure marker

    Procedure:
    1. Find marker with most adjacent 1s (structure marker if clear winner)
    2. Non-structure markers do simultaneous BFS through 0s only
    3. Color 1s: all with structure color OR each by most frequent adjacent color
    4. Spread to fill any remaining cells
    """
    from collections import deque, Counter

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    result = [[None] * cols for _ in range(rows)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Find all markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] > 1:
                markers.append((r, c, grid[r][c]))

    # Find marker with most adjacent 1s
    adjacent_counts = []
    for mr, mc, color in markers:
        count = sum(
            1 for dr, dc in directions
            if 0 <= mr + dr < rows and 0 <= mc + dc < cols and grid[mr + dr][mc + dc] == 1
        )
        adjacent_counts.append((count, color))

    adjacent_counts.sort(reverse=True)

    # Check for clear structure marker (uniquely most adjacent 1s)
    structure_color = None
    if adjacent_counts and adjacent_counts[0][0] > 0:
        if len(adjacent_counts) == 1 or adjacent_counts[0][0] > adjacent_counts[1][0]:
            structure_color = adjacent_counts[0][1]

    # Separate markers
    if structure_color:
        region_markers = [(r, c, color) for r, c, color in markers if color != structure_color]
    else:
        region_markers = markers

    # Phase 1: Simultaneous BFS from region markers through 0s only
    queue = deque()
    for mr, mc, color in region_markers:
        result[mr][mc] = color
        queue.append((mr, mc, color))

    while queue:
        r, c, color = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] is None:
                if grid[nr][nc] == 0:
                    result[nr][nc] = color
                    queue.append((nr, nc, color))

    # Phase 2: Color 1s
    if structure_color is not None:
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    result[r][c] = structure_color
    else:
        # Color 1s based on most frequent adjacent color
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    adjacent_colors = []
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] is not None:
                            adjacent_colors.append(result[nr][nc])

                    if adjacent_colors:
                        # Use most common color, with min value as tiebreaker
                        counter = Counter(adjacent_colors)
                        result[r][c] = counter.most_common(1)[0][0]

    # Phase 3: Fill remaining cells
    queue2 = deque()
    for r in range(rows):
        for c in range(cols):
            if result[r][c] is not None:
                queue2.append((r, c))

    while queue2:
        r, c = queue2.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] is None:
                result[nr][nc] = result[r][c]
                queue2.append((nr, nc))

    return result
