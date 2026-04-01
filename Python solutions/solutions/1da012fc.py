def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a rectangular region filled with 5s
    2. Within the 5s region, there are marker cells with non-5 values
    3. Outside the 5s region, there are connected components of another value
    4. Each component (size > 1) gets recolored with a marker value
    5. Matching is based on reading order (top-to-bottom, left-to-right)

    Procedure:
    1. Find the 5s region bounding box
    2. Extract marker values from the 5s region (non-5, non-0 values)
    3. Find connected components outside the 5s region (size > 1 only)
    4. Sort both markers and components by reading order
    5. Recolor each component with its corresponding marker value
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find the 5-region boundaries
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    # Extract colors from the 5-region (non-5, non-0 values)
    colors = []
    color_positions = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            if grid[r][c] != 5 and grid[r][c] != 0:
                colors.append(grid[r][c])
                color_positions.append((r, c))

    # Sort colors by their position (row first, then column)
    color_data = list(zip(colors, color_positions))
    color_data.sort(key=lambda x: (x[1][0], x[1][1]))
    colors = [x[0] for x in color_data]

    # Find the target value (non-0, non-5 value outside the 5-region)
    target_value = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5:
                # Check if it's outside the 5-region
                if not (min_row <= r <= max_row and min_col <= c <= max_col):
                    target_value = grid[r][c]
                    break
        if target_value is not None:
            break

    # Find all connected components of the target value
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, component):
        if (
            r < 0
            or r >= rows
            or c < 0
            or c >= cols
            or visited[r][c]
            or grid[r][c] != target_value
        ):
            return
        visited[r][c] = True
        component.append((r, c))
        # 4-directional connectivity
        dfs(r + 1, c, component)
        dfs(r - 1, c, component)
        dfs(r, c + 1, component)
        dfs(r, c - 1, component)

    for r in range(rows):
        for c in range(cols):
            if (
                grid[r][c] == target_value
                and not visited[r][c]
                and not (min_row <= r <= max_row and min_col <= c <= max_col)
            ):
                component = []
                dfs(r, c, component)
                # Only include components with more than 1 cell
                if len(component) > 1:
                    components.append(component)

    # Sort components by their top-left position
    def get_top_left(component):
        return min(component, key=lambda x: (x[0], x[1]))

    components.sort(key=get_top_left)

    # Assign colors to components and update result
    for i, component in enumerate(components):
        color = colors[i % len(colors)]
        for r, c in component:
            result[r][c] = color

    return result
