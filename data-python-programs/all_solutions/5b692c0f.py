def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains colored regions (non-zero values) with 4s marking axes of symmetry
    2. Output makes each region symmetric around its axis/axes
    3. Multiple separate components are processed independently
    4. When reflecting, we OR values at symmetric positions
    5. The 4s themselves are also made symmetric

    Procedure:
    1. Find connected components of non-zero values
    2. For each component, identify axis/axes marked by 4s
    3. Apply horizontal reflection (around vertical axis) and/or vertical reflection (around horizontal axis)
    4. OR values at symmetric positions, keeping 4s when present
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * cols for _ in range(rows)]

    def bfs(start_r, start_c):
        """Find connected component starting from (start_r, start_c)"""
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return component

    def process_component(component):
        """Process a single connected component"""
        if not component:
            return

        # Find bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        # Find lines of 4s
        four_rows = {}
        four_cols = {}

        for r, c in component:
            if grid[r][c] == 4:
                four_rows[r] = four_rows.get(r, 0) + 1
                four_cols[c] = four_cols.get(c, 0) + 1

        # Identify primary axes (lines with 3+ 4s)
        h_axis = None  # horizontal axis (row number) for vertical reflection
        v_axis = None  # vertical axis (column number) for horizontal reflection

        for r, count in four_rows.items():
            if count >= 3:
                h_axis = r
                break

        for c, count in four_cols.items():
            if count >= 3:
                v_axis = c
                break

        # Extend bounding box to include reflections
        if v_axis is not None:
            # Extend horizontally
            dist_left = v_axis - min_c
            dist_right = max_c - v_axis
            max_dist = max(dist_left, dist_right)
            min_c = max(0, v_axis - max_dist)
            max_c = min(cols - 1, v_axis + max_dist)

        if h_axis is not None:
            # Extend vertically
            dist_up = h_axis - min_r
            dist_down = max_r - h_axis
            max_dist = max(dist_up, dist_down)
            min_r = max(0, h_axis - max_dist)
            max_r = min(rows - 1, h_axis + max_dist)

        # Find rows that contain 4s at the axis
        axis_rows = []
        if v_axis is not None and 0 <= v_axis < cols:
            axis_rows = [r for r in range(min_r, max_r + 1) if grid[r][v_axis] == 4]
        if axis_rows:
            min_axis_row = min(axis_rows)
            max_axis_row = max(axis_rows)
        else:
            min_axis_row = max_axis_row = None

        # Apply reflections
        if v_axis is not None:  # Horizontal reflection around vertical axis
            for r in range(min_r, max_r + 1):
                # Check if this row contains the axis (has a 4 at the axis column)
                row_has_axis = (grid[r][v_axis] == 4 if 0 <= v_axis < cols else False)

                # Check if row is below the axis region (for directional copy rule)
                below_axis_region = False
                if axis_rows and not row_has_axis and r > max_axis_row:
                    dist_to_axis = r - max_axis_row
                    below_axis_region = (dist_to_axis <= 2)

                # Count non-zero cells on left vs right to determine strategy
                left_count = sum(1 for c in range(min_c, v_axis) if result[r][c] != 0 and result[r][c] != 4)
                right_count = sum(1 for c in range(v_axis + 1, max_c + 1) if result[r][c] != 0 and result[r][c] != 4)

                # Use intersection if left has more cells than right AND right has <= 50% of left's cells
                if left_count > 0 and right_count > 0:
                    use_intersection = row_has_axis and left_count > right_count and right_count <= left_count / 2
                else:
                    use_intersection = False

                for c in range(min_c, max_c + 1):
                    mirror_c = 2 * v_axis - c
                    if 0 <= mirror_c < cols and c <= mirror_c:  # Process each pair once
                        val1 = result[r][c]
                        val2 = result[r][mirror_c]

                        # For rows without the axis marker
                        if not row_has_axis:
                            # Don't reflect cells immediately adjacent to axis
                            if abs(c - v_axis) == 1 or abs(mirror_c - v_axis) == 1:
                                # Only keep if both sides already have the same non-zero value
                                if val1 != val2:
                                    result[r][c] = 0
                                    result[r][mirror_c] = 0
                                    continue
                            # For rows below the axis region, only copy from left to right (not right to left)
                            elif below_axis_region and c < v_axis < mirror_c and val1 == 0 and val2 != 0:
                                # Don't copy from right to left
                                result[r][c] = 0
                                result[r][mirror_c] = 0
                                continue

                        # For rows with axis and imbalanced sides, use AND (intersection)
                        if use_intersection and c < v_axis < mirror_c:
                            # Only keep if both sides have non-zero
                            if val1 == 4 or val2 == 4:
                                combined = 4
                            elif val1 != 0 and val2 != 0:
                                combined = val1 if val1 != 4 else val2
                            else:
                                combined = 0
                            result[r][c] = combined
                            result[r][mirror_c] = combined
                        else:
                            # OR the values, with 4 taking priority
                            if val1 == 4 or val2 == 4:
                                combined = 4
                            elif val1 != 0:
                                combined = val1
                            elif val2 != 0:
                                combined = val2
                            else:
                                combined = 0
                            result[r][c] = combined
                            result[r][mirror_c] = combined

        if h_axis is not None:  # Vertical reflection around horizontal axis
            for r in range(min_r, max_r + 1):
                mirror_r = 2 * h_axis - r
                if 0 <= mirror_r < rows and r <= mirror_r:  # Process each pair once
                    for c in range(min_c, max_c + 1):
                        # Check if this column contains the axis (has a 4 at the axis row)
                        col_has_axis = (grid[h_axis][c] == 4 if 0 <= h_axis < rows else False)

                        val1 = result[r][c]
                        val2 = result[mirror_r][c]

                        # For columns without the axis marker, don't reflect cells immediately adjacent to axis
                        if not col_has_axis and (abs(r - h_axis) == 1 or abs(mirror_r - h_axis) == 1):
                            # Only keep if both sides already have the same non-zero value
                            if val1 != val2:
                                result[r][c] = 0
                                result[mirror_r][c] = 0
                                continue

                        # For horizontal axis, always use OR (symmetric reflection)
                        # OR the values, with 4 taking priority
                        if val1 == 4 or val2 == 4:
                            combined = 4
                        elif val1 != 0:
                            combined = val1
                        elif val2 != 0:
                            combined = val2
                        else:
                            combined = 0
                        result[r][c] = combined
                        result[mirror_r][c] = combined

    # Find and process all connected components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                component = bfs(r, c)
                process_component(component)

    return result
