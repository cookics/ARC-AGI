def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid filled mostly with 7s (background color) and some non-7 colored regions.
    2. The output transforms the grid by adding 2s at specific border positions around rectangular regions.
    3. Only rectangular regions of non-7 colors that meet size criteria get bordered with 2s.
    4. A rectangular region can be either solid (completely filled) or hollow (only the perimeter).

    Procedure:
    1. Find all connected components of non-7 colors using flood fill algorithm.
    2. For each component, check if it forms a valid rectangle (solid or hollow) with sufficient size.
    3. If it does, place 2s at specific border positions around the rectangle boundary.
    4. Return the modified grid with 2s placed at appropriate border locations.
    """

    def is_rectangle(positions):
        """Check if positions form a rectangle (solid or hollow) and is significant enough"""
        if not positions:
            return False

        # Get bounding rectangle
        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)

        # Only consider rectangles that are significant enough
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # Rule: must be square (2x2, 3x3, etc.) or both dimensions >= 3
        if height != width and (height < 3 or width < 3):
            return False

        if height < 2 or width < 2:
            return False

        pos_set = set(positions)

        # Check if it's a solid rectangle
        expected_solid = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                expected_solid.add((r, c))

        if pos_set == expected_solid:
            return True

        # Check if it's a hollow rectangle (only perimeter)
        expected_hollow = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    expected_hollow.add((r, c))

        if pos_set == expected_hollow:
            return True

        return False

    def add_border(grid, positions, rows, cols):
        """Add 2s at specific positions around the rectangular region"""
        if not positions:
            return

        # Get bounding rectangle
        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)

        # Border positions to place 2s
        border_positions = [
            (min_r - 1, min_c),
            (min_r - 1, max_c),  # Above corners
            (max_r + 1, min_c),
            (max_r + 1, max_c),  # Below corners
            (min_r, min_c - 1),
            (max_r, min_c - 1),  # Left of edges
            (min_r, max_c + 1),
            (max_r, max_c + 1),  # Right of edges
        ]

        for r, c in border_positions:
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 7:
                grid[r][c] = 2

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all non-7 colors and their positions
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7:
                color = grid[r][c]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((r, c))

    # For each color, find connected components
    for color, positions in color_positions.items():
        if not positions:
            continue

        visited = set()
        components = []

        for r, c in positions:
            if (r, c) not in visited:
                component = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    visited.add((cr, cc))
                    component.append((cr, cc))

                    # Check 4-connected neighbors
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and (nr, nc) not in visited
                            and grid[nr][nc] == color
                        ):
                            stack.append((nr, nc))

                components.append(component)

        # For each component, check if it forms a rectangle
        for component in components:
            if is_rectangle(component):
                add_border(result, component, rows, cols)

    return result
