def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Looking at the examples, the pattern is:
    1. Find all positions with value 1 in the input
    2. Draw lines (horizontal and vertical) through these positions
    3. Any rectangular regions of 2s that intersect with these lines become regions of 1s
    4. The lines themselves are drawn as 1s throughout the grid

    From example 1:
    - 1s at (0,6) and (9,6) -> vertical line at column 6
    - Region of 2s at (2-5, 3-6) intersects column 6 -> becomes 1s
    - Vertical line of 1s drawn through column 6

    From example 2:
    - 1s create lines at row 8 and column 7
    - 2s regions intersecting these lines become 1s

    Hand-validate: The transformation draws crossing lines through 1s
    and converts intersecting 2s regions to 1s.

    Procedure:
    1. Find all 1s in input and identify line positions
    2. Copy input to result
    3. Draw lines of 1s through the identified rows/columns
    4. Convert 2s regions that intersect with lines to 1s
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy input

    # Find all positions with value 1
    ones_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones_positions.append((r, c))

    if not ones_positions:
        return result

    # Find lines between aligned pairs of 1s
    lines = set()

    # Check all pairs of 1s for alignment
    for i in range(len(ones_positions)):
        for j in range(i + 1, len(ones_positions)):
            r1, c1 = ones_positions[i]
            r2, c2 = ones_positions[j]

            if r1 == r2:  # Same row - horizontal line
                start_c, end_c = min(c1, c2), max(c1, c2)
                for c in range(start_c, end_c + 1):
                    lines.add((r1, c))
            elif c1 == c2:  # Same column - vertical line
                start_r, end_r = min(r1, r2), max(r1, r2)
                for r in range(start_r, end_r + 1):
                    lines.add((r, c1))

    # Draw the lines
    for r, c in lines:
        result[r][c] = 1

    # Find rectangular regions of 2s and convert them to 1s if they intersect lines
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill_region(start_r, start_c):
        """Find connected region of 2s and check if it intersects with lines"""
        if visited[start_r][start_c] or grid[start_r][start_c] != 2:
            return []

        region = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= cols
                or visited[r][c]
                or grid[r][c] != 2
            ):
                continue

            visited[r][c] = True
            region.append((r, c))

            # Add 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))

        return region

    # Process all 2s regions
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                region = flood_fill_region(r, c)

                if region:
                    # Check if this region intersects with any lines
                    intersects_line = False
                    for rr, cc in region:
                        if (rr, cc) in lines:
                            intersects_line = True
                            break

                    # If region intersects with lines, convert it to 1s
                    if intersects_line:
                        for rr, cc in region:
                            result[rr][cc] = 1

    return result
