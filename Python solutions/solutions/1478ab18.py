def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an 8x8 grid with background color 7 and marker color 5
    2. Output fills a polygon defined by the 5s with color 8
    3. The 5s mark vertices of a polygon to be filled
    4. Bottom-right corner 5 (if present) is excluded from polygon
    5. Interior cells become 8, keeping vertex 5s unchanged

    Procedure:
    1. Find all cells with value 5 (exclude bottom-right corner)
    2. Compute convex hull of vertices
    3. Use proper point-in-polygon test to fill interior with 8s
    """
    from copy import deepcopy

    rows, cols = len(grid), len(grid[0])
    result = deepcopy(grid)

    # Find all 5s (vertices), excluding bottom-right corner if it's a 5
    vertices = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                # Exclude 5 at bottom-right corner
                if not (r == rows - 1 and c == cols - 1):
                    vertices.append((r, c))

    if len(vertices) < 3:
        return result

    # Compute convex hull using Graham scan
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def convex_hull(points):
        points = sorted(points)
        if len(points) <= 1:
            return points

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    hull = convex_hull(vertices)

    # Helper function to check if point is inside polygon using winding number
    def point_inside_polygon(px, py, poly):
        """Check if point (px,py) is inside polygon using ray casting"""
        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or px <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # Fill interior of hull
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 5:  # Don't overwrite vertices
                # Use (c, r) for x, y in standard coordinates
                if point_inside_polygon(r, c, hull):
                    result[r][c] = 8

    return result
