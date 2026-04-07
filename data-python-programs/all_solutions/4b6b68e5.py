def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with frames/boundaries made of specific colors
    2. Output fills enclosed regions with the most common non-zero, non-frame color found inside
    3. Each frame color creates an enclosed region that gets filled separately
    4. Frame colors (even incomplete ones) are preserved; scattered hint colors are removed

    Procedure:
    1. Identify frame colors (those forming connected lines) vs hint colors (isolated cells)
    2. For each frame color, find and fill enclosed regions
    3. Keep all frame colors in result; remove hint colors outside enclosed regions
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])

    # Find all distinct non-zero colors
    all_colors = set()
    for row in grid:
        for val in row:
            if val != 0:
                all_colors.add(val)

    # Identify frame colors: colors that form connected structures (lines/boundaries)
    # A color forms a structure if any of its cells has neighbors of the same color
    frame_colors = set()
    for color in all_colors:
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    # Check if this cell has any neighbor with same color
                    has_neighbor = False
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == color:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        frame_colors.add(color)
                        break
            if color in frame_colors:
                break

    # Start with only frame colors
    result = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] in frame_colors:
                result[r][c] = grid[r][c]

    # Now fill enclosed regions for each frame color
    for frame_color in frame_colors:
        # Find cells enclosed by this frame color using flood fill from exterior
        visited = [[False] * cols for _ in range(rows)]

        def flood_fill(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return
            if visited[r][c]:
                return
            if grid[r][c] == frame_color:  # Frame acts as barrier
                return

            visited[r][c] = True
            flood_fill(r + 1, c)
            flood_fill(r - 1, c)
            flood_fill(r, c + 1)
            flood_fill(r, c - 1)

        # Start flood fill from all borders
        for r in range(rows):
            flood_fill(r, 0)
            flood_fill(r, cols - 1)
        for c in range(cols):
            flood_fill(0, c)
            flood_fill(rows - 1, c)

        # Find enclosed cells (not visited and not frame)
        enclosed = []
        for r in range(rows):
            for c in range(cols):
                if not visited[r][c] and grid[r][c] != frame_color:
                    enclosed.append((r, c))

        if not enclosed:
            continue

        # Count non-zero, non-frame colors in enclosed region
        color_count = {}
        for r, c in enclosed:
            val = grid[r][c]
            if val != 0 and val != frame_color:
                color_count[val] = color_count.get(val, 0) + 1

        if not color_count:
            continue

        # Find most common color
        fill_color = max(color_count, key=color_count.get)

        # Fill all enclosed cells with the most common color
        for r, c in enclosed:
            result[r][c] = fill_color

    return result
