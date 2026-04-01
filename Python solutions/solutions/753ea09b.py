def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 3 colors: background (most frequent), filler (middle), marker (least)
    2. Each filler cell fills all background cells to its left in the same row
    3. Filler cells also propagate to vertical/diagonal neighbors forming connected regions
    4. Markers remain unchanged

    Procedure:
    1. Identify colors by frequency
    2. For each filler cell, fill leftward in its row
    3. Use BFS to propagate filler to adjacent cells vertically/diagonally
    4. Preserve markers throughout
    """
    from collections import Counter, deque

    rows = len(grid)
    cols = len(grid[0])

    # Count color frequencies
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    color_counts = Counter(all_values)
    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

    background = sorted_colors[0][0]
    filler = sorted_colors[1][0]
    marker = sorted_colors[2][0]

    result = [row[:] for row in grid]

    # Step 1: Fill leftward from each filler cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == filler:
                # Fill all background cells to the left
                for left_c in range(c):
                    if result[r][left_c] == background:
                        result[r][left_c] = filler

    # Step 2: BFS to expand to diagonally connected cells (limited expansion)
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == filler:
                queue.append((r, c))

    processed = set()
    while queue:
        r, c = queue.popleft()
        if (r, c) in processed:
            continue
        processed.add((r, c))

        # Only expand to 3 specific neighbors (diagonal/vertical in adjacent rows)
        for dr in [-1, 1]:  # Only up and down
            for dc in [-1, 0, 1]:  # Diagonal left, straight, diagonal right
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if result[nr][nc] == background:
                        result[nr][nc] = filler
                        queue.append((nr, nc))

    return result
