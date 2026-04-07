def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a line of 1s (vertical, horizontal, or diagonal)
    2. The line acts as a mirror/reflection axis
    3. Colored blocks (non-8, non-1 values) may be reflected across the line
    4. Only colors that have at least one connected component forming EXACTLY a 2x2 rectangle get reflected
    5. When a color qualifies, ALL cells of that color are reflected

    Procedure:
    1. Find the line of 1s and determine its type (vertical/horizontal/diagonal)
    2. For each color, find all connected components
    3. Check if any component is EXACTLY a 2x2 filled rectangle
    4. If yes, reflect all cells of that color across the line
    """

    def get_line_type(ones):
        """Determine the type of line formed by 1s"""
        if not ones:
            return None, None

        # Check vertical line (same column)
        cols = [c for r, c in ones]
        if len(set(cols)) == 1:
            return "vertical", cols[0]

        # Check horizontal line (same row)
        rows = [r for r, c in ones]
        if len(set(rows)) == 1:
            return "horizontal", rows[0]

        # Check diagonal line (row + col = k)
        sums = [r + c for r, c in ones]
        if len(set(sums)) == 1:
            return "diagonal_plus", sums[0]

        # Check diagonal line (row - col = k)
        diffs = [r - c for r, c in ones]
        if len(set(diffs)) == 1:
            return "diagonal_minus", diffs[0]

        return None, None

    def get_connected_components(grid, color):
        """Find all connected components of a given color"""
        rows, cols = len(grid), len(grid[0])
        visited = set()
        components = []

        def bfs(start_r, start_c):
            component = set()
            queue = [(start_r, start_c)]
            visited.add((start_r, start_c))
            component.add((start_r, start_c))

            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        (nr, nc) not in visited and grid[nr][nc] == color):
                        visited.add((nr, nc))
                        component.add((nr, nc))
                        queue.append((nr, nc))

            return component

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    component = bfs(r, c)
                    components.append(component)

        return components

    def is_exactly_2x2(component):
        """Check if a component forms exactly a 2x2 filled rectangle"""
        if len(component) != 4:
            return False

        # Find bounding box
        rows = [r for r, c in component]
        cols = [c for r, c in component]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        # Must be 2x2
        if max_r - min_r != 1 or max_c - min_c != 1:
            return False

        # All 4 cells must be present
        expected = {(min_r, min_c), (min_r, min_c + 1),
                    (min_r + 1, min_c), (min_r + 1, min_c + 1)}
        return component == expected

    def has_exactly_2x2_component(grid, color):
        """Check if a color has any component that is exactly 2x2"""
        components = get_connected_components(grid, color)
        return any(is_exactly_2x2(comp) for comp in components)

    def reflect_point(r, c, line_type, line_param):
        """Reflect a point across the line"""
        if line_type == "vertical":
            return (r, 2 * line_param - c)
        elif line_type == "horizontal":
            return (2 * line_param - r, c)
        elif line_type == "diagonal_plus":
            return (line_param - c, line_param - r)
        elif line_type == "diagonal_minus":
            return (c + line_param, r - line_param)
        return None

    def is_valid_position(pos, rows, cols):
        """Check if position is within grid bounds"""
        r, c = pos
        return 0 <= r < rows and 0 <= c < cols

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find line of 1s
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]
    if not ones:
        return result

    # Determine line type and parameters
    line_type, line_param = get_line_type(ones)
    if line_type is None:
        return result

    # Find all colors (excluding background 8 and line 1)
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in {8, 1}:
                colors.add(grid[r][c])

    # For each color, check if it has exactly 2x2 component
    for color in colors:
        if has_exactly_2x2_component(grid, color):
            # Reflect all cells of this color
            cells_to_reflect = [
                (r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color
            ]
            for r, c in cells_to_reflect:
                reflected_pos = reflect_point(r, c, line_type, line_param)
                if reflected_pos and is_valid_position(reflected_pos, rows, cols):
                    rr, rc = reflected_pos
                    result[rr][rc] = color

    return result
