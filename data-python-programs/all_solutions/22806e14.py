def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a cross/plus pattern (5 cells forming a plus shape)
    2. The cross color acts as a marker
    3. All connected components of non-background, non-marker colors are identified
    4. Components with odd-sized bounding boxes get their center marked with the marker color
    5. The cross pattern is removed from the output

    Procedure:
    1. Find the cross pattern and extract marker color
    2. Find all connected components of all non-background colors (excluding the cross itself)
    3. For components with odd bounding box dimensions, mark center with marker color
    4. Remove the cross pattern
    """
    from collections import deque
    import copy

    rows = len(grid)
    cols = len(grid[0])
    result = copy.deepcopy(grid)

    # Find cross pattern
    cross_center = None
    marker_color = None

    for r in range(1, rows-1):
        for c in range(1, cols-1):
            val = grid[r][c]
            if val != 7:  # not background
                # Check if this is center of a cross
                if (grid[r-1][c] == val and grid[r+1][c] == val and
                    grid[r][c-1] == val and grid[r][c+1] == val):
                    # Check it's not part of a larger block (diagonals should be different)
                    if (grid[r-1][c-1] != val or grid[r-1][c+1] != val or
                        grid[r+1][c-1] != val or grid[r+1][c+1] != val):
                        cross_center = (r, c)
                        marker_color = val
                        break
        if cross_center:
            break

    if not cross_center:
        return result

    # Find all connected components
    visited = [[False] * cols for _ in range(rows)]

    # Mark the cross cells as visited so we don't process them
    cr, cc = cross_center
    cross_cells = [(cr, cc), (cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)]
    for r, c in cross_cells:
        visited[r][c] = True

    def bfs(start_r, start_c, color):
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                    if grid[nr][nc] == color:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

        return component

    # Find all unique colors (excluding background)
    all_colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7:
                all_colors.add(grid[r][c])

    # Find minimum non-marker color
    other_colors = [c for c in all_colors if c != marker_color]
    min_other_color = min(other_colors) if other_colors else None

    # Process all non-background components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7 and not visited[r][c]:
                component = bfs(r, c, grid[r][c])

                # Find bounding box
                min_r = min(cell[0] for cell in component)
                max_r = max(cell[0] for cell in component)
                min_c = min(cell[1] for cell in component)
                max_c = max(cell[1] for cell in component)

                height = max_r - min_r + 1
                width = max_c - min_c + 1

                # If both dimensions are odd, mark center
                if height % 2 == 1 and width % 2 == 1:
                    center_r = (min_r + max_r) // 2
                    center_c = (min_c + max_c) // 2
                    result[center_r][center_c] = marker_color

    # Remove the cross only if marker < all other colors
    if min_other_color is not None and marker_color < min_other_color:
        for r, c in cross_cells:
            result[r][c] = 7

    return result
