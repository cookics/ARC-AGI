def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Looking at the training examples, the simplest pattern is:
    - Two points of the same color that are aligned (same row/column) get connected with lines
    Let me start with this basic pattern and build up from there.

    Procedure:
    1. Find all points by color
    2. For colors with exactly 2 points that are aligned, connect them with lines
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all non-zero points grouped by color
    points_by_color = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in points_by_color:
                    points_by_color[color] = []
                points_by_color[color].append((r, c))

    # Rule 1: Linear connections for 2 points of same color
    for color, points in points_by_color.items():
        if len(points) == 2:
            r1, c1 = points[0]
            r2, c2 = points[1]

            if r1 == r2:  # Same row - horizontal line
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    result[r1][c] = color
            elif c1 == c2:  # Same column - vertical line
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    result[r][c1] = color

    return result
