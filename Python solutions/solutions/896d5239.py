def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid contains 0s, 1s, and 3s marking polygon vertices
    2. Output fills polygon interiors with 8s using scanline algorithm
    3. Use convex hull to get ordered polygon vertices
    4. Scanline fill: for each row, find edge intersections and fill between

    Procedure:
    1. Find and cluster nearby 3s
    2. Compute convex hull for each cluster
    3. Use scanline algorithm to fill polygon interior
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all 3s
    threes = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 3]
    if not threes:
        return result

    # Cluster using BFS with distance threshold
    visited = set()
    clusters = []

    def find_cluster(start):
        cluster = []
        queue = [start]
        visited.add(start)
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for cand in threes:
                if cand not in visited:
                    r1, c1 = curr
                    r2, c2 = cand
                    dist_sq = (r1 - r2) ** 2 + (c1 - c2) ** 2
                    if dist_sq <= 10:  # Distance <= sqrt(10) ≈ 3.16
                        visited.add(cand)
                        queue.append(cand)
        return cluster

    for pos in threes:
        if pos not in visited:
            comp = find_cluster(pos)
            if len(comp) >= 3:
                clusters.append(comp)

    # Convex hull
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def convex_hull(points):
        points = sorted(set(points))
        if len(points) <= 2:
            return points

        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        return lower[:-1] + upper[:-1]

    # Scanline fill
    def fill_polygon_scanline(hull):
        if len(hull) < 3:
            return

        min_row = min(p[0] for p in hull)
        max_row = max(p[0] for p in hull)

        # Group vertices by row for quick lookup
        vertices_by_row = {}
        for r, c in hull:
            if r not in vertices_by_row:
                vertices_by_row[r] = []
            vertices_by_row[r].append(c)

        for row in range(min_row, max_row + 1):
            intersections = []

            # Find all edge intersections with this scanline
            for i in range(len(hull)):
                p1r, p1c = hull[i]
                p2r, p2c = hull[(i + 1) % len(hull)]

                if p1r == p2r:  # Horizontal edge
                    continue

                # Ensure p1r < p2r
                if p1r > p2r:
                    p1r, p1c, p2r, p2c = p2r, p2c, p1r, p1c

                # Include vertices at endpoints
                if p1r <= row <= p2r:
                    # Compute intersection column
                    if p1r == p2r:
                        # Should not happen (horizontal edge)
                        continue
                    t = (row - p1r) / (p2r - p1r)
                    col = p1c + t * (p2c - p1c)
                    intersections.append(col)

            if len(intersections) >= 2:
                intersections.sort()

                # Remove duplicates (vertices cause duplicate intersections)
                unique_intersections = []
                for col in intersections:
                    if not unique_intersections or abs(col - unique_intersections[-1]) > 0.01:
                        unique_intersections.append(col)

                # Fill between pairs of intersections
                if len(unique_intersections) >= 2:
                    for i in range(0, len(unique_intersections) - 1, 2):
                        start_col = int(unique_intersections[i] + 0.5)
                        end_col = int(unique_intersections[i + 1] + 0.5)

                        for col in range(start_col, end_col + 1):
                            if 0 <= col < len(grid[0]) and grid[row][col] != 3:
                                result[row][col] = 8

    # Process each cluster
    for comp in clusters:
        # Order points by angle from centroid (like polar sort)
        import math
        centroid_r = sum(r for r, c in comp) / len(comp)
        centroid_c = sum(c for r, c in comp) / len(comp)

        def angle_from_centroid(p):
            return math.atan2(p[1] - centroid_c, p[0] - centroid_r)

        ordered = sorted(comp, key=angle_from_centroid)
        fill_polygon_scanline(ordered)

    return result
