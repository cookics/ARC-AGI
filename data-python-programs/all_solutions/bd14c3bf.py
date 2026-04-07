def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with values 0, 1, and 2
    2. Output converts certain 1-patterns to 2s
    3. The template (2s) defines what kind of frames to convert
    4. If template has 2 edges filled: convert frames with 2 opposite edges + middle row/col filled
    5. If template has 3+ edges filled: convert frames with 3+ edges filled

    Procedure:
    1. Analyze template to determine edge count
    2. Find all connected components of 1s
    3. For each component, check if it's a valid frame based on template
    4. Convert valid frames from 1s to 2s
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    # Analyze template (2s) to determine rule
    template_cells = [
        (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2
    ]
    template_edge_count = 0

    if template_cells:
        min_r = min(r for r, c in template_cells)
        max_r = max(r for r, c in template_cells)
        min_c = min(c for r, c in template_cells)
        max_c = max(c for r, c in template_cells)

        # Count edges in template
        if all(grid[min_r][c] == 2 for c in range(min_c, max_c + 1)):
            template_edge_count += 1
        if all(grid[max_r][c] == 2 for c in range(min_c, max_c + 1)):
            template_edge_count += 1
        if all(grid[r][min_c] == 2 for r in range(min_r, max_r + 1)):
            template_edge_count += 1
        if all(grid[r][max_c] == 2 for r in range(min_r, max_r + 1)):
            template_edge_count += 1

    def dfs(r, c, component):
        """Find all cells in a connected component of 1s"""
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 1:
            return
        visited[r][c] = True
        component.append((r, c))
        dfs(r + 1, c, component)
        dfs(r - 1, c, component)
        dfs(r, c + 1, component)
        dfs(r, c - 1, component)

    def is_valid_frame(component):
        """Check if component is a valid frame to convert"""
        if not component:
            return False

        # Get bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        # Check all 4 corners are 1s
        if grid[min_r][min_c] != 1 or grid[min_r][max_c] != 1:
            return False
        if grid[max_r][min_c] != 1 or grid[max_r][max_c] != 1:
            return False

        # Count filled edges
        first_row_filled = all(grid[min_r][c] == 1 for c in range(min_c, max_c + 1))
        last_row_filled = all(grid[max_r][c] == 1 for c in range(min_c, max_c + 1))
        first_col_filled = all(grid[r][min_c] == 1 for r in range(min_r, max_r + 1))
        last_col_filled = all(grid[r][max_c] == 1 for r in range(min_r, max_r + 1))

        edge_count = sum(
            [first_row_filled, last_row_filled, first_col_filled, last_col_filled]
        )

        # Apply rule based on template
        if template_edge_count == 4:
            # Template has 4 edges: require pattern to have exactly 4 edges AND hollow interior
            if edge_count == 4:
                # Check interior is empty
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        if grid[r][c] == 1:
                            return False
                return True
            return False
        elif template_edge_count == 3:
            # Template has 3 edges: convert if pattern has >= 3 edges AND hollow interior
            if edge_count >= 3:
                # Check interior is empty
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        if grid[r][c] == 1:
                            return False
                return True
            return False
        else:
            # Template has <= 2 edges: convert if 2 opposite edges filled AND middle row/col filled
            if first_row_filled and last_row_filled:
                mid_c = (min_c + max_c) // 2
                if all(grid[r][mid_c] == 1 for r in range(min_r, max_r + 1)):
                    return True
            if first_col_filled and last_col_filled:
                mid_r = (min_r + max_r) // 2
                if all(grid[mid_r][c] == 1 for c in range(min_c, max_c + 1)):
                    return True
            return False

    # Find all connected components of 1s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                component = []
                dfs(r, c, component)

                if component and is_valid_frame(component):
                    for cr, cc in component:
                        result[cr][cc] = 2

    return result
