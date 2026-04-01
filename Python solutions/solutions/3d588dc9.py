def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains connected components of 0s surrounded by 7s (background)
    2. Some components contain colored shapes (non-0, non-7 values) like diagonal/triangular patterns
    3. For certain 0s components, we need to:
       - Find the largest rectangular core
       - Remove non-rectangular extensions (convert to 7)
       - Convert one edge column of the rectangle to 6 (the edge closest to a large colored shape)
    4. Very small 0s components (1-2 cells) are completely removed

    Procedure:
    1. Find all connected components of 0s
    2. Find the largest colored shape (non-0, non-7 component with many cells)
    3. For each 0s component that overlaps row-wise with this colored shape:
       - Find largest rectangle within the component
       - Remove cells outside rectangle
       - Convert edge closest to colored shape to 6
    4. Remove very small 0s components
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    def find_component(r, c, value):
        """BFS to find connected component of given value"""
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != value:
            return []

        visited[r][c] = True
        component = [(r, c)]
        queue = [(r, c)]

        while queue:
            cr, cc = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == value:
                    visited[nr][nc] = True
                    component.append((nr, nc))
                    queue.append((nr, nc))

        return component

    # Find all components
    all_components = []
    visited = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c]:
                comp = find_component(r, c, grid[r][c])
                if comp:
                    all_components.append((grid[r][c], comp))

    # Find largest colored shape (non-0, non-7, with >5 cells)
    colored_shapes = [(val, cells) for val, cells in all_components if val not in [0, 7] and len(cells) > 5]
    if not colored_shapes:
        return result

    largest_shape = max(colored_shapes, key=lambda x: len(x[1]))
    shape_rows = set(r for r, c in largest_shape[1])
    shape_col_min = min(c for r, c in largest_shape[1])
    shape_col_max = max(c for r, c in largest_shape[1])

    # Find 0s components
    zeros_components = [(cells) for val, cells in all_components if val == 0]

    # Process each 0s component
    for component in zeros_components:
        comp_rows = set(r for r, c in component)

        # Check if this component overlaps row-wise with largest colored shape
        if not (comp_rows & shape_rows):
            continue  # Leave this component unchanged

        # For small overlapping components, also skip (don't modify)
        if len(component) <= 2:
            continue

        # Find largest rectangle in this component
        row_list = sorted(comp_rows)
        cols_by_row = {}
        for r, c in component:
            if r not in cols_by_row:
                cols_by_row[r] = []
            cols_by_row[r].append(c)

        for r in cols_by_row:
            cols_by_row[r].sort()

        # Find best rectangle by trying all row ranges
        best_rect = None
        best_area = 0

        for i in range(len(row_list)):
            for j in range(i, len(row_list)):
                row_range = row_list[i:j+1]
                # Find common column range across these rows
                col_min = max(min(cols_by_row[r]) for r in row_range)
                col_max = min(max(cols_by_row[r]) for r in row_range)

                if col_min <= col_max:
                    # Verify all cells in rectangle are 0s
                    valid = all(grid[r][c] == 0 for r in row_range for c in range(col_min, col_max + 1))
                    if valid:
                        area = (col_max - col_min + 1) * len(row_range)
                        if area > best_area:
                            best_area = area
                            best_rect = (row_range, col_min, col_max)

        if not best_rect:
            continue

        rect_rows, rect_col_min, rect_col_max = best_rect

        # Remove cells outside rectangle
        for r, c in component:
            if r not in rect_rows or c < rect_col_min or c > rect_col_max:
                result[r][c] = 7

        # Determine which edge to convert to 6
        # If colored shape is to the left, convert left edge; if to right, convert right edge
        if shape_col_max < rect_col_min:
            # Shape is to the left
            edge_col = rect_col_min
        else:
            # Shape is to the right (or overlapping)
            edge_col = rect_col_max

        # Convert edge to 6
        for r in rect_rows:
            result[r][edge_col] = 6

    return result
