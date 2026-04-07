def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large grid with multiple scattered clusters of colored cells
    2. Output is a specific bounding box from the input
    3. Need to identify which component/bbox to return

    Procedure:
    1. Find all connected components using BFS (4-connectivity)
    2. Extract bounding box for each component
    3. Return a specific one based on some criteria
    """
    from collections import deque

    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    def bfs(start_r, start_c, color):
        """Find connected component starting from (start_r, start_c)"""
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return component

    # Find all connected components
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                color = grid[r][c]
                component = bfs(r, c, color)

                # Find bounding box
                min_r = min(cell[0] for cell in component)
                max_r = max(cell[0] for cell in component)
                min_c = min(cell[1] for cell in component)
                max_c = max(cell[1] for cell in component)

                # Extract bounding box as a grid
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                bbox = [[0] * width for _ in range(height)]

                for cell_r, cell_c in component:
                    bbox[cell_r - min_r][cell_c - min_c] = color

                components.append({
                    'bbox': bbox,
                    'size': len(component),
                    'top_left': (min_r, min_c),
                    'bbox_size': (height, width)
                })

    if not components:
        return [[0]]

    # Try: return the bottom-most component (highest row value)
    bottom_comp = max(components, key=lambda c: c['top_left'][0])
    return bottom_comp['bbox']
