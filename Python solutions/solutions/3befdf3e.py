def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing rectangular regions with non-zero values.
    2. Each rectangle has a border color (outer ring) and an inner color (center area).
    3. The output transforms each rectangle by swapping and extending these color regions.
    4. The inner color expands to fill the entire original rectangle area.
    5. The border color fills the original inner area.
    6. The border color extends outward from the original rectangle by (rectangle_width - 2) cells.

    Procedure:
    1. Find all connected rectangular regions in the input grid.
    2. For each rectangle, identify the border color and inner color.
    3. Fill the entire rectangle area with the inner color.
    4. Fill the original inner area with the border color.
    5. Extend the border color outward by (rectangle_width - 2) cells in all directions.
    6. Return the transformed grid.
    """

    def find_connected_rectangles(grid):
        rectangles = []
        visited = [[False] * len(grid[0]) for _ in range(len(grid))]

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] != 0 and not visited[r][c]:
                    component = get_connected_component(grid, r, c, visited)
                    rect = analyze_component(grid, component)
                    if rect:
                        rectangles.append(rect)

        return rectangles

    def get_connected_component(grid, start_r, start_c, visited):
        component = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            if (
                0 <= r < len(grid)
                and 0 <= c < len(grid[0])
                and not visited[r][c]
                and grid[r][c] != 0
            ):
                visited[r][c] = True
                component.append((r, c))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((r + dr, c + dc))

        return component

    def analyze_component(grid, component):
        if not component:
            return None

        min_r = min(pos[0] for pos in component)
        max_r = max(pos[0] for pos in component)
        min_c = min(pos[1] for pos in component)
        max_c = max(pos[1] for pos in component)

        expected_positions = set()
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                expected_positions.add((r, c))

        if set(component) != expected_positions:
            return None

        border_color = grid[min_r][min_c]

        inner_color = None
        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if grid[r][c] != border_color:
                    inner_color = grid[r][c]
                    break
            if inner_color is not None:
                break

        return {
            "min_r": min_r,
            "max_r": max_r,
            "min_c": min_c,
            "max_c": max_c,
            "border_color": border_color,
            "inner_color": inner_color,
        }

    def apply_transformation(grid, rect):
        if rect is None:
            return

        min_r, max_r = rect["min_r"], rect["max_r"]
        min_c, max_c = rect["min_c"], rect["max_c"]
        border_color = rect["border_color"]
        inner_color = rect["inner_color"]

        rect_width = max_c - min_c + 1
        extension = rect_width - 2

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                grid[r][c] = inner_color

        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                grid[r][c] = border_color

        for i in range(extension):
            target_r = min_r - 1 - i
            if 0 <= target_r < len(grid):
                for c in range(min_c, max_c + 1):
                    grid[target_r][c] = border_color
            target_r = max_r + 1 + i
            if 0 <= target_r < len(grid):
                for c in range(min_c, max_c + 1):
                    grid[target_r][c] = border_color

        for i in range(extension):
            target_c = min_c - 1 - i
            if 0 <= target_c < len(grid[0]):
                for r in range(min_r, max_r + 1):
                    grid[r][target_c] = border_color
            target_c = max_c + 1 + i
            if 0 <= target_c < len(grid[0]):
                for r in range(min_r, max_r + 1):
                    grid[r][target_c] = border_color

    result = [row[:] for row in grid]

    rectangles = find_connected_rectangles(grid)
    for rect in rectangles:
        apply_transformation(result, rect)

    return result
