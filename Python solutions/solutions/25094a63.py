def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid of integers (values 1-8)
    2. Output is the same grid with certain rectangular regions replaced with 4s
    3. The rectangles are completely filled with a single uniform value
    4. The rectangles are relatively large (at least 5x5, minimum area ~25 cells)
    5. Multiple non-overlapping rectangles can exist with different values

    Procedure:
    1. For each unique value in the grid, find all maximal rectangles filled with that value
    2. Use a greedy approach: iteratively find the largest rectangle, mark it, repeat
    3. Filter rectangles by size (area >= 25, height >= 5, width >= 5)
    4. Replace the values in these rectangles with 4s
    """

    def find_largest_rectangle_at(grid, r, c, value, marked):
        """Find the largest rectangle starting at position (r, c)."""
        rows = len(grid)
        cols = len(grid[0])

        # Find the maximum width at row r
        max_width = 0
        for c2 in range(c, cols):
            if not marked[r][c2] and grid[r][c2] == value:
                max_width += 1
            else:
                break

        if max_width == 0:
            return None

        # Extend downward, maintaining a rectangular shape
        height = 1
        width = max_width
        for r2 in range(r + 1, rows):
            # Check if all cells in this row (for the current width) have the value
            row_width = 0
            for c2 in range(c, c + width):
                if c2 >= cols or marked[r2][c2] or grid[r2][c2] != value:
                    break
                row_width += 1

            if row_width == width:
                height += 1
            else:
                break

        return (r, c, height, width)

    def find_all_maximal_rectangles(grid, value):
        """Find all maximal rectangles filled with the given value."""
        rows = len(grid)
        cols = len(grid[0])
        rectangles = []
        marked = [[False] * cols for _ in range(rows)]

        # Greedy approach: find largest rectangle iteratively
        while True:
            best_rect = None
            best_area = 0

            # Try all possible starting positions
            for r in range(rows):
                for c in range(cols):
                    if marked[r][c] or grid[r][c] != value:
                        continue

                    rect = find_largest_rectangle_at(grid, r, c, value, marked)
                    if rect:
                        _, _, height, width = rect
                        area = height * width
                        if area > best_area:
                            best_area = area
                            best_rect = rect

            # Stop if no valid rectangle found
            if best_rect is None or best_area < 25:
                break

            r, c, height, width = best_rect
            # Check minimum dimensions
            if height >= 5 and width >= 5:
                rectangles.append(best_rect)
                # Mark cells as used
                for rr in range(r, r + height):
                    for cc in range(c, c + width):
                        marked[rr][cc] = True

        return rectangles

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Get all unique values
    values = set()
    for row in grid:
        values.update(row)

    # For each value, find maximal rectangles
    for value in values:
        rectangles = find_all_maximal_rectangles(grid, value)

        # Replace with 4s
        for r, c, h, w in rectangles:
            for rr in range(r, r + h):
                for cc in range(c, c + w):
                    result[rr][cc] = 4

    return result
