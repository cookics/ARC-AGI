def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has exactly 2 non-background (non-7) colors
    2. One color forms a line pattern (vertical, horizontal, or diagonal)
    3. Other color forms discrete shapes (crosses, X patterns) at multiple positions
    4. Transformation:
       - Shape patterns move to the line's endpoints
       - Line color creates a new pattern based on shape positions
    5. Example 1: diagonal line → crosses at diagonal endpoints + triangular pattern
    6. Example 2: vertical line → horizontal line (rotation with marker swap)
    7. Example 3: vertical line → diamond pattern expanding from middle

    Procedure:
    1. Identify the two colors and their positions
    2. Determine which is the line and which forms shapes
    3. Extract line endpoints and shape centers
    4. Place shapes at line endpoints
    5. Create new pattern with line color based on line type and shape positions
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [[7] * cols for _ in range(rows)]

    # Find all non-7 positions grouped by color
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7:
                val = grid[r][c]
                if val not in color_positions:
                    color_positions[val] = []
                color_positions[val].append((r, c))

    if len(color_positions) != 2:
        return result

    colors = list(color_positions.keys())
    color1, color2 = colors[0], colors[1]
    pos1, pos2 = color_positions[color1], color_positions[color2]

    # Determine which is the line (more connected/aligned) vs shapes (clustered)
    def analyze_pattern(positions):
        rows_set = set(r for r, c in positions)
        cols_set = set(c for r, c in positions)

        # Check for vertical line
        if len(cols_set) == 1 and len(rows_set) > 2:
            return "vertical_line"
        # Check for horizontal line
        if len(rows_set) == 1 and len(cols_set) > 2:
            return "horizontal_line"
        # Check for diagonal
        if len(positions) >= 4:
            sorted_pos = sorted(positions)
            diffs = [(sorted_pos[i+1][0] - sorted_pos[i][0],
                     sorted_pos[i+1][1] - sorted_pos[i][1])
                    for i in range(len(sorted_pos)-1)]
            # Check if roughly diagonal
            if all(abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0) for dr, dc in diffs):
                return "diagonal_line"
        return "shapes"

    type1 = analyze_pattern(pos1)
    type2 = analyze_pattern(pos2)

    # Identify line and shapes
    if "line" in type1:
        line_type, line_color, line_pos = type1, color1, pos1
        shape_color, shape_pos = color2, pos2
    else:
        line_type, line_color, line_pos = type2, color2, pos2
        shape_color, shape_pos = color1, pos1

    # Extract shape structure from first component
    from collections import deque

    def find_components(positions):
        pos_set = set(positions)
        visited = set()
        components = []

        for start_pos in positions:
            if start_pos in visited:
                continue

            component = []
            queue = deque([start_pos])
            visited.add(start_pos)

            while queue:
                r, c = queue.popleft()
                component.append((r, c))

                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in pos_set and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

            components.append(component)

        return components

    def get_shape_structure(component):
        if not component:
            return [], (0, 0)
        # Find center of the component
        center_r = sum(r for r, c in component) // len(component)
        center_c = sum(c for r, c in component) // len(component)
        structure = [(r - center_r, c - center_c) for r, c in component]
        return structure, (center_r, center_c)

    shape_components = find_components(shape_pos)
    shape_structure, original_center = get_shape_structure(shape_components[0]) if shape_components else ([], (0, 0))

    # Find line endpoints
    line_pos_sorted = sorted(line_pos)
    line_start = line_pos_sorted[0]
    line_end = line_pos_sorted[-1]

    # Place shapes at line endpoints
    for center in [line_start, line_end]:
        for dr, dc in shape_structure:
            nr, nc = center[0] + dr, center[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result[nr][nc] = shape_color

    # Create new pattern based on line type and number of shape components
    if line_type == "vertical_line" and len(shape_components) == 2:
        # Vertical line + 2 markers → Horizontal line (simple rotation)
        shape_rows = sorted(set(r for r, c in shape_pos))
        shape_cols = sorted(set(c for r, c in shape_pos))

        if len(shape_cols) >= 2:
            # Draw horizontal line at middle shape row
            target_row = shape_rows[len(shape_rows)//2] if shape_rows else line_start[0]
            for c in range(min(shape_cols), max(shape_cols) + 1):
                result[target_row][c] = line_color

    elif line_type == "horizontal_line":
        # Horizontal → Vertical
        shape_rows = sorted(set(r for r, c in shape_pos))
        shape_cols = sorted(set(c for r, c in shape_pos))

        if len(shape_rows) >= 2:
            # Draw vertical line at middle shape column
            target_col = shape_cols[len(shape_cols)//2] if shape_cols else line_start[1]
            for r in range(min(shape_rows), max(shape_rows) + 1):
                result[r][target_col] = line_color

    elif line_type == "diagonal_line":
        # Diagonal → V-shaped pattern (two diagonal lines expanding)
        # Find shape centroids/key positions
        shape_centers = []
        for comp in shape_components:
            center_r = sum(r for r, c in comp) // len(comp)
            center_c = sum(c for r, c in comp) // len(comp)
            shape_centers.append((center_r, center_c))

        shape_centers.sort()

        if len(shape_centers) >= 2:
            # Start from first shape center, draw two diagonal lines toward other centers
            start_r, start_c = shape_centers[0]
            end_rows = [r for r, c in shape_centers[1:]]
            end_cols = [c for r, c in shape_centers[1:]]

            col_min = min(end_cols) if end_cols else start_c - 6
            col_max = max(end_cols) if end_cols else start_c

            # Calculate target row (distance to expand)
            target_row = start_r + (col_max - col_min) if col_max != col_min else start_r + 6

            # Draw left diagonal (going down-left)
            for i in range(target_row - start_r + 1):
                r = start_r + i
                c = start_c - i
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = line_color

            # Draw right diagonal/vertical (going down or down-right)
            for i in range(target_row - start_r + 1):
                r = start_r + i
                c = start_c
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = line_color

            # Fill bottom row between the two diagonals
            if target_row < rows:
                left_c = start_c - (target_row - start_r)
                right_c = start_c
                for c in range(max(0, left_c), min(cols, right_c + 1)):
                    result[target_row][c] = line_color
    else:
        # Shapes as line → Diamond pattern based on shape centers
        # Find shape centers
        shape_centers = []
        for comp in shape_components:
            center_r = sum(r for r, c in comp) // len(comp)
            center_c = sum(c for r, c in comp) // len(comp)
            shape_centers.append((center_r, center_c))

        shape_centers.sort()

        if len(shape_centers) >= 3:
            # Three or more shapes - create diamond outline from first to last, expanding to middle
            start_r, start_c = shape_centers[0]
            mid_r, mid_c = shape_centers[len(shape_centers)//2]
            end_r, end_c = shape_centers[-1]

            # Draw diamond as two diagonal lines (left stays fixed, right expands/contracts)
            for r in range(start_r, end_r + 1):
                # Left edge always at start_c
                result[r][start_c] = line_color

                # Right edge position
                if r <= mid_r:
                    # Expanding phase
                    offset = r - start_r
                else:
                    # Contracting phase
                    offset = end_r - r

                right_c = start_c + offset
                if right_c != start_c and 0 <= right_c < cols:
                    result[r][right_c] = line_color
        elif len(shape_centers) == 2:
            # Two shapes - simple expanding pattern
            start_r, start_c = shape_centers[0]
            end_r, end_c = shape_centers[1]

            for r in range(start_r, end_r + 1):
                progress = (r - start_r) / max(1, end_r - start_r)
                width = 1 + int(progress * abs(end_c - start_c))

                for i in range(width):
                    c = min(start_c, end_c) + i
                    if 0 <= c < cols:
                        result[r][c] = line_color

    return result
