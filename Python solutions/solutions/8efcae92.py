def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 20x20 grid containing multiple rectangular regions of 1s and 2s on a background of 0s
    2. Output is a single rectangular region extracted from the input
    3. Multiple non-zero rectangular components exist, separated by 0 values
    4. The target region is selected based on position and size criteria

    Procedure:
    1. Find all connected components of non-zero values using flood fill
    2. For each component, compute its bounding rectangle and area
    3. Select the bottommost rectangle if its area >= 30, otherwise select the largest rectangle
    4. Extract and return the selected rectangular region
    """

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill(start_row, start_col):
        """Find all cells in the connected component starting from (start_row, start_col)"""
        if (
            start_row < 0
            or start_row >= rows
            or start_col < 0
            or start_col >= cols
            or visited[start_row][start_col]
            or grid[start_row][start_col] == 0
        ):
            return []

        cells = []
        stack = [(start_row, start_col)]

        while stack:
            r, c = stack.pop()
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= cols
                or visited[r][c]
                or grid[r][c] == 0
            ):
                continue

            visited[r][c] = True
            cells.append((r, c))

            # Add adjacent cells
            stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])

        return cells

    def get_bounding_rectangle(cells):
        """Get the bounding rectangle for a set of cells"""
        if not cells:
            return None

        min_row = min(r for r, c in cells)
        max_row = max(r for r, c in cells)
        min_col = min(c for r, c in cells)
        max_col = max(c for r, c in cells)

        return (min_row, min_col, max_row, max_col)

    # Find all connected components and their bounding rectangles
    rectangles = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = flood_fill(r, c)
                if cells:
                    rect = get_bounding_rectangle(cells)
                    if rect:
                        area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
                        rectangles.append((rect, area))

    # Select the largest rectangle by area
    assert len(rectangles) > 0, "No rectangular regions found"

    # Try bottommost rectangle first (highest starting row)
    rectangles_by_position = sorted(rectangles, key=lambda x: x[0][0], reverse=True)
    bottommost_rect = rectangles_by_position[0]

    # Try largest rectangle
    rectangles_by_area = sorted(rectangles, key=lambda x: x[1], reverse=True)
    largest_rect = rectangles_by_area[0]

    # If bottommost is reasonably large (area >= 30), choose it; otherwise choose largest
    if bottommost_rect[1] >= 30:
        selected_rect = bottommost_rect[0]
    else:
        selected_rect = largest_rect[0]

    # Extract the selected rectangle
    min_row, min_col, max_row, max_col = selected_rect
    result = []
    for r in range(min_row, max_row + 1):
        row = []
        for c in range(min_col, max_col + 1):
            row.append(grid[r][c])
        result.append(row)

    return result
