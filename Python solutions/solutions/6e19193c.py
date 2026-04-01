def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with L-shaped connected components of non-zero values
    2. Output is the same grid with added diagonal lines (slope -1, row+col=constant)
    3. Diagonal lines extend from the elbow point of each L-shaped component
    4. The elbow is the corner where horizontal and vertical segments meet
    5. Direction of diagonal extension depends on the gap between components

    Procedure:
    1. Find all connected components of non-zero values using DFS
    2. For each component, identify the elbow point (has both horizontal and vertical neighbors)
    3. Calculate the gap between components (vertical distance between them)
    4. If gap > 1, diagonals extend toward each other; otherwise away from each other
    5. For each elbow at (r,c), draw diagonal line with equation row+col=r+c in appropriate direction
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the original grid

    # Find all non-zero points
    non_zero_points = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                non_zero_points.append((r, c))

    # Find connected components using DFS
    visited = set()
    components = []

    def dfs(r, c, component, value):
        if (
            (r, c) in visited
            or r < 0
            or r >= rows
            or c < 0
            or c >= cols
            or grid[r][c] != value
        ):
            return
        visited.add((r, c))
        component.append((r, c))
        # Check 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, component, value)

    for r, c in non_zero_points:
        if (r, c) not in visited:
            component = []
            dfs(r, c, component, grid[r][c])
            if component:
                components.append(component)

    # Find elbows and component bounding boxes
    elbows = []
    component_bounds = []

    for component in components:
        # Find component bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        component_bounds.append((min_r, max_r, min_c, max_c))

        # Find the elbow: the point that has neighbors in both horizontal and vertical directions
        for r, c in component:
            has_horizontal_neighbor = False
            has_vertical_neighbor = False

            # Check for horizontal neighbors
            if (r, c - 1) in component or (r, c + 1) in component:
                has_horizontal_neighbor = True

            # Check for vertical neighbors
            if (r - 1, c) in component or (r + 1, c) in component:
                has_vertical_neighbor = True

            # The elbow has both horizontal and vertical neighbors
            if has_horizontal_neighbor and has_vertical_neighbor:
                elbows.append((r, c, grid[r][c]))
                break

    # Sort elbows and bounds by row to match them up
    paired = list(zip(elbows, component_bounds))
    paired.sort(key=lambda x: x[0][0])  # Sort by elbow row
    elbows, component_bounds = zip(*paired)

    # Calculate gap between components to determine direction
    gap = (
        component_bounds[1][0] - component_bounds[0][1] - 1
    )  # min_r of second - max_r of first - 1

    # Draw diagonal lines from each elbow
    for i, ((elbow_r, elbow_c, value), (min_r, max_r, min_c, max_c)) in enumerate(
        zip(elbows, component_bounds)
    ):
        diagonal_sum = elbow_r + elbow_c

        if gap > 1:  # Large gap: diagonals extend toward each other
            if i == 0:  # First elbow extends down-left
                start_row = max_r + 1
                for r in range(start_row, rows):
                    c = diagonal_sum - r
                    if 0 <= c < cols and grid[r][c] == 0:
                        result[r][c] = value
            else:  # Second elbow extends up-right
                end_row = min_r - 1
                for r in range(end_row, -1, -1):
                    c = diagonal_sum - r
                    if 0 <= c < cols and grid[r][c] == 0:
                        result[r][c] = value
        else:  # Small/no gap: diagonals extend away from each other
            if i == 0:  # First elbow extends up-right
                end_row = min_r - 1
                for r in range(end_row, -1, -1):
                    c = diagonal_sum - r
                    if 0 <= c < cols and grid[r][c] == 0:
                        result[r][c] = value
            else:  # Second elbow extends down-left
                start_row = max_r + 1
                for r in range(start_row, rows):
                    c = diagonal_sum - r
                    if 0 <= c < cols and grid[r][c] == 0:
                        result[r][c] = value

    return result
