def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing values 0, 2, and other colors.
    2. The output shows rectangles outlined by 2s with their interiors filled.
    3. Rectangles with interior area 9 (3x3) are filled with color 8.
    4. Rectangles with interior area 25 (5x5) are filled with color 4.
    5. Rectangles with interior area 49 (7x7) are filled with color 3.
    6. Any existing 2s inside the rectangles are preserved in the output.

    Procedure:
    1. Find all rectangles outlined by 2s in the grid.
    2. For each rectangle, calculate the interior bounds and area.
    3. Determine the fill color based on the interior area.
    4. Fill the interior with the assigned color while preserving existing 2s.
    """

    result = [row[:] for row in grid]  # Deep copy
    rows, cols = len(grid), len(grid[0])

    def find_rectangles():
        rectangles = []
        visited = [[False] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2 and not visited[r][c]:
                    # Try to find a rectangle starting from this point
                    rect = get_rectangle_bounds(r, c, visited)
                    if rect:
                        rectangles.append(rect)

        return rectangles

    def get_rectangle_bounds(start_r, start_c, visited):
        # Find the complete rectangular border
        if grid[start_r][start_c] != 2:
            return None

        # Find all connected 2s using BFS/DFS
        stack = [(start_r, start_c)]
        border_cells = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in border_cells or r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != 2:
                continue

            border_cells.add((r, c))
            stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])

        if not border_cells:
            return None

        # Find bounding box
        min_r = min(r for r, c in border_cells)
        max_r = max(r for r, c in border_cells)
        min_c = min(c for r, c in border_cells)
        max_c = max(c for r, c in border_cells)

        # Verify it's a proper rectangle border
        expected_border = set()
        for r in range(min_r, max_r + 1):
            expected_border.add((r, min_c))
            expected_border.add((r, max_c))
        for c in range(min_c, max_c + 1):
            expected_border.add((min_r, c))
            expected_border.add((max_r, c))

        # Check if the found border matches expected rectangle
        if border_cells == expected_border:
            # Mark as visited
            for r, c in border_cells:
                visited[r][c] = True
            return (min_r, min_c, max_r, max_c)

        return None

    def get_color_for_area(area):
        if area == 9:
            return 8
        elif area == 25:
            return 4
        elif area == 49:
            return 3
        else:
            return 1  # Default fallback

    # Find all rectangles and fill them
    rectangles = find_rectangles()

    for min_r, min_c, max_r, max_c in rectangles:
        # Calculate interior area and get color
        interior_height = max_r - min_r - 1
        interior_width = max_c - min_c - 1
        interior_area = interior_height * interior_width

        if interior_area > 0:
            fill_color = get_color_for_area(interior_area)

            # Fill interior, preserving existing 2s
            for rr in range(min_r + 1, max_r):
                for cc in range(min_c + 1, max_c):
                    if grid[rr][cc] != 2:  # Preserve existing 2s
                        result[rr][cc] = fill_color

    return result
