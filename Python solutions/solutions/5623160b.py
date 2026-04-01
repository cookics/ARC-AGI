def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid where 7 is the background color
    2. Components with value 9 or 0 stay fixed (anchors)
    3. Other components move to edges based on their position relative to nearest 9-component
    4. Movement direction is determined by bounding box overlap with 9-component

    Procedure:
    1. Find all connected components using 4-connectivity
    2. Identify 9 and 0 components as fixed anchors
    3. For each moving component:
       - Find nearest 9-component by centroid distance
       - Determine direction based on bounding box overlap:
         * Rows overlap, columns don't: move horizontally (left/right)
         * Columns overlap, rows don't: move vertically (up/down)
         * Both overlap: move vertically based on position in 9's row range
       - Move component to the appropriate edge
    4. Construct output grid with all components
    """
    from collections import deque

    height = len(grid)
    width = len(grid[0])

    # Find connected components using BFS
    visited = [[False] * width for _ in range(height)]
    components = []

    def bfs(start_r, start_c):
        value = grid[start_r][start_c]
        cells = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            cells.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr][nc] and grid[nr][nc] == value:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return value, cells

    for r in range(height):
        for c in range(width):
            if not visited[r][c] and grid[r][c] != 7:
                value, cells = bfs(r, c)
                components.append((value, cells))

    # Separate fixed and moving components
    fixed_components = []
    moving_components = []
    nine_components = []

    for value, cells in components:
        if value == 9 or value == 0:
            fixed_components.append((value, cells))
            if value == 9:
                nine_components.append((value, cells))
        else:
            moving_components.append((value, cells))

    # Helper functions
    def get_bbox(cells):
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        return min(rows), max(rows), min(cols), max(cols)

    def get_centroid(cells):
        return sum(r for r, c in cells) / len(cells), sum(c for r, c in cells) / len(cells)

    def distance(c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    # Create output grid
    result = [[7] * width for _ in range(height)]

    # Place fixed components
    for value, cells in fixed_components:
        for r, c in cells:
            result[r][c] = value

    # Move and place moving components
    for value, cells in moving_components:
        centroid = get_centroid(cells)

        # Find nearest 9-component
        nearest_nine = None
        min_dist = float('inf')
        for nine_value, nine_cells in nine_components:
            nine_centroid = get_centroid(nine_cells)
            dist = distance(centroid, nine_centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_nine = nine_cells

        # Get bounding boxes
        comp_min_r, comp_max_r, comp_min_c, comp_max_c = get_bbox(cells)
        nine_min_r, nine_max_r, nine_min_c, nine_max_c = get_bbox(nearest_nine)

        # Determine movement direction based on overlap
        rows_overlap = not (comp_max_r < nine_min_r or comp_min_r > nine_max_r)
        cols_overlap = not (comp_max_c < nine_min_c or comp_min_c > nine_max_c)

        if rows_overlap and not cols_overlap:
            # Move horizontally
            if comp_max_c < nine_min_c:
                # Move left to edge
                offset_r, offset_c = 0, -comp_min_c
            else:
                # Move right to edge
                offset_r, offset_c = 0, (width - 1) - comp_max_c
        elif cols_overlap and not rows_overlap:
            # Move vertically
            if comp_max_r < nine_min_r:
                # Move up to edge
                offset_r, offset_c = -comp_min_r, 0
            else:
                # Move down to edge
                offset_r, offset_c = (height - 1) - comp_max_r, 0
        elif rows_overlap and cols_overlap:
            # Both overlap: move vertically based on position
            nine_mid_r = (nine_min_r + nine_max_r) / 2
            comp_centroid_r = centroid[0]
            if comp_centroid_r < nine_mid_r:
                # Upper half: move up
                offset_r, offset_c = -comp_min_r, 0
            else:
                # Lower half: move down
                offset_r, offset_c = (height - 1) - comp_max_r, 0
        else:
            # Neither overlap: use centroid difference
            nine_centroid = get_centroid(nearest_nine)
            delta_r = centroid[0] - nine_centroid[0]
            delta_c = centroid[1] - nine_centroid[1]
            if abs(delta_r) > abs(delta_c):
                # Move vertically
                if delta_r < 0:
                    offset_r, offset_c = -comp_min_r, 0
                else:
                    offset_r, offset_c = (height - 1) - comp_max_r, 0
            else:
                # Move horizontally
                if delta_c < 0:
                    offset_r, offset_c = 0, -comp_min_c
                else:
                    offset_r, offset_c = 0, (width - 1) - comp_max_c

        # Apply offset and place component
        for r, c in cells:
            new_r, new_c = r + offset_r, c + offset_c
            result[new_r][new_c] = value

    return result
