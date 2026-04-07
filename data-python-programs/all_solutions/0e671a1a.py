def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has exactly three colored cells (values 2, 3, 4) on a grid of 0s
    2. Output connects these three points with paths of 5s using rectilinear geometry
    3. Connection pattern depends on point configuration:
       - If top-left corner exists: star from that corner
       - If median column has a point: vertical trunk through it
       - Otherwise: chain connection through middle point

    Procedure:
    1. Find the three colored points
    2. Determine connection strategy based on geometry
    3. Draw paths with value 5
    4. Restore original point values
    """

    # Find the three colored points
    points = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] in [2, 3, 4]:
                points.append((r, c, grid[r][c]))

    if len(points) != 3:
        return grid

    result = [row[:] for row in grid]

    # Sort points by row
    points_sorted = sorted(points, key=lambda p: p[0])
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]

    # Find bounding box
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Check if there's a point at top-left corner
    corner_point = None
    for p in points:
        if p[0] == min_r and p[1] == min_c:
            corner_point = p
            break

    # Calculate median column centeredness
    cols_sorted = sorted(cols)
    median_col = cols_sorted[1]
    col_range = max(cols) - min(cols)
    median_distance_from_min = median_col - min(cols)
    median_ratio = median_distance_from_min / col_range if col_range > 0 else 0

    # Decide strategy based on geometry
    if corner_point and median_ratio >= 0.35:
        # Star pattern from corner
        for p in points:
            if p == corner_point:
                continue
            # Draw L-path: horizontal first if column distance is larger
            if abs(p[1] - corner_point[1]) >= abs(p[0] - corner_point[0]):
                # Horizontal then vertical
                for c in range(min(corner_point[1], p[1]), max(corner_point[1], p[1]) + 1):
                    if result[corner_point[0]][c] == 0:
                        result[corner_point[0]][c] = 5
                for r in range(min(corner_point[0], p[0]), max(corner_point[0], p[0]) + 1):
                    if result[r][p[1]] == 0:
                        result[r][p[1]] = 5
            else:
                # Vertical then horizontal
                for r in range(min(corner_point[0], p[0]), max(corner_point[0], p[0]) + 1):
                    if result[r][corner_point[1]] == 0:
                        result[r][corner_point[1]] = 5
                for c in range(min(corner_point[1], p[1]), max(corner_point[1], p[1]) + 1):
                    if result[p[0]][c] == 0:
                        result[p[0]][c] = 5
    elif corner_point and median_ratio < 0.35:
        # Chain strategy when corner exists but median is off-center
        middle_point = points_sorted[1]

        # Connect top to middle
        p_top = points_sorted[0]
        if abs(p_top[1] - middle_point[1]) >= abs(p_top[0] - middle_point[0]):
            # Horizontal first
            for c in range(min(p_top[1], middle_point[1]), max(p_top[1], middle_point[1]) + 1):
                if result[p_top[0]][c] == 0:
                    result[p_top[0]][c] = 5
            for r in range(min(p_top[0], middle_point[0]), max(p_top[0], middle_point[0]) + 1):
                if result[r][middle_point[1]] == 0:
                    result[r][middle_point[1]] = 5
        else:
            # Vertical first
            for r in range(min(p_top[0], middle_point[0]), max(p_top[0], middle_point[0]) + 1):
                if result[r][p_top[1]] == 0:
                    result[r][p_top[1]] = 5
            for c in range(min(p_top[1], middle_point[1]), max(p_top[1], middle_point[1]) + 1):
                if result[middle_point[0]][c] == 0:
                    result[middle_point[0]][c] = 5

        # Connect middle to bottom
        p_bottom = points_sorted[2]
        if abs(middle_point[1] - p_bottom[1]) >= abs(middle_point[0] - p_bottom[0]):
            # Horizontal first
            for c in range(min(middle_point[1], p_bottom[1]), max(middle_point[1], p_bottom[1]) + 1):
                if result[middle_point[0]][c] == 0:
                    result[middle_point[0]][c] = 5
            for r in range(min(middle_point[0], p_bottom[0]), max(middle_point[0], p_bottom[0]) + 1):
                if result[r][p_bottom[1]] == 0:
                    result[r][p_bottom[1]] = 5
        else:
            # Vertical first
            for r in range(min(middle_point[0], p_bottom[0]), max(middle_point[0], p_bottom[0]) + 1):
                if result[r][middle_point[1]] == 0:
                    result[r][middle_point[1]] = 5
            for c in range(min(middle_point[1], p_bottom[1]), max(middle_point[1], p_bottom[1]) + 1):
                if result[p_bottom[0]][c] == 0:
                    result[p_bottom[0]][c] = 5
    else:
        # Check if any point is at median column
        median_point = None
        for p in points:
            if p[1] == median_col:
                median_point = p
                break

        if median_point:
            # Vertical trunk strategy
            # Draw vertical line through median column
            for r in range(min_r, max_r + 1):
                if result[r][median_col] == 0:
                    result[r][median_col] = 5

            # Connect each point horizontally to the trunk
            for p in points:
                if p[1] != median_col:
                    for c in range(min(p[1], median_col), max(p[1], median_col) + 1):
                        if result[p[0]][c] == 0:
                            result[p[0]][c] = 5

            # If two points are on opposite sides of trunk, connect them with rectangle
            other_points = [p for p in points if p[1] != median_col]
            if len(other_points) == 2:
                p1, p2 = other_points
                if (p1[1] < median_col < p2[1]) or (p2[1] < median_col < p1[1]):
                    # Determine which point is on the left and which on top
                    if p1[1] < p2[1]:
                        left_point, right_point = p1, p2
                    else:
                        left_point, right_point = p2, p1

                    if p1[0] < p2[0]:
                        top_point, bottom_point = p1, p2
                    else:
                        top_point, bottom_point = p2, p1

                    # Check if right point is close to trunk (adjacent or near)
                    if right_point[1] - median_col <= 2:
                        # Draw vertical at right column connecting the two rows
                        for r in range(top_point[0], bottom_point[0] + 1):
                            if result[r][right_point[1]] == 0:
                                result[r][right_point[1]] = 5
                    else:
                        # Draw horizontal at top row from left col to right col
                        for c in range(left_point[1], right_point[1] + 1):
                            if result[top_point[0]][c] == 0:
                                result[top_point[0]][c] = 5

                        # Draw vertical at left column from top row to bottom row
                        for r in range(top_point[0], bottom_point[0] + 1):
                            if result[r][left_point[1]] == 0:
                                result[r][left_point[1]] = 5

    # Restore original colored cells
    for p in points:
        result[p[0]][p[1]] = p[2]

    return result
