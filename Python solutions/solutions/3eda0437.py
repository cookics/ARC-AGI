def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Looking at the expected vs actual, I need to find the LARGEST rectangular region
    of zeros that can be found across multiple rows. Let me try a different approach:
    find all possible rectangular regions and pick the largest ones.



    Procedure:
    1. Find all rectangular regions of zeros that span at least 2 rows
    2. For each valid rectangle found, replace zeros in those regions with 6s
    3. Use a visited array to avoid overlapping rectangles
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the grid
    visited = [[False] * cols for _ in range(rows)]

    def find_largest_rectangle(start_r, start_c):
        """Find the largest rectangle of zeros starting from (start_r, start_c)"""
        if grid[start_r][start_c] != 0 or visited[start_r][start_c]:
            return None

        # Try different widths and find the one that gives the largest rectangle
        best_rect = None
        best_area = 0

        # Find maximum possible width from start position
        max_possible_width = 0
        for c in range(start_c, cols):
            if grid[start_r][c] == 0:
                max_possible_width += 1
            else:
                break

        # Try each width from 3 to max_possible_width
        for width in range(3, max_possible_width + 1):
            # Find maximum height with this width
            height = 1
            for r in range(start_r + 1, rows):
                # Check if this row has zeros for the full width
                valid = True
                for c in range(start_c, start_c + width):
                    if grid[r][c] != 0:
                        valid = False
                        break
                if valid:
                    height += 1
                else:
                    break

            # If we have at least height 2, this is a valid rectangle
            if height >= 2:
                area = width * height
                if area > best_area:
                    best_area = area
                    best_rect = (
                        start_r,
                        start_c,
                        start_r + height - 1,
                        start_c + width - 1,
                    )

        return best_rect

    def mark_visited(r1, c1, r2, c2):
        """Mark rectangle as visited"""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                visited[r][c] = True

    def fill_rectangle(r1, c1, r2, c2):
        """Fill rectangle with 6s"""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                result[r][c] = 6

    # Find all rectangles, prioritizing by area
    rectangles = []

    for r in range(rows):
        for c in range(cols):
            rect = find_largest_rectangle(r, c)
            if rect:
                r1, c1, r2, c2 = rect
                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                rectangles.append((area, r1, c1, r2, c2))

    # Sort by area (descending) and process largest first
    rectangles.sort(reverse=True)

    for area, r1, c1, r2, c2 in rectangles:
        # Check if any part is already visited
        overlap = False
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if visited[r][c]:
                    overlap = True
                    break
            if overlap:
                break

        if not overlap:
            mark_visited(r1, c1, r2, c2)
            fill_rectangle(r1, c1, r2, c2)

    return result
