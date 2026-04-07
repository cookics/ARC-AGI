def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a large 2D grid (20+ x 20+ cells) containing multiple distinct rectangular objects composed of various colored values (1-7).
    2. The output is a much smaller binary grid where 8 marks positions with differences and 0 marks positions with similarities between two selected objects.
    3. The task is to identify two similar rectangular objects within the input grid and highlight their differences through element-wise comparison.
    4. Objects are detected as 8-connected non-zero regions that form meaningful rectangular patterns with sufficient size and density.

    Procedure:
    1. Detect all connected non-zero regions using 8-connectivity and extract their rectangular bounding boxes.
    2. Score each object based on area, cell density, and structural features (presence of key values like 4, 6, 7).
    3. Select the two highest-scoring non-overlapping objects for comparison.
    4. Normalize objects to the same dimensions and perform element-wise comparison.
    5. Output only rows containing differences and meaningful content.
    """

    rows, cols = len(grid), len(grid[0])

    # Find all non-zero rectangular regions
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    def get_object_bounds(start_r, start_c):
        """Get tight bounding box for connected non-zero region"""
        min_r = max_r = start_r
        min_c = max_c = start_c
        queue = [(start_r, start_c)]
        region_cells = set()

        while queue:
            r, c = queue.pop(0)
            if (r, c) in region_cells or r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] == 0:
                continue

            visited[r][c] = True
            region_cells.add((r, c))
            min_r, max_r = min(min_r, r), max(max_r, r)
            min_c, max_c = min(min_c, c), max(max_c, c)

            # Explore 8-connected neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    queue.append((r + dr, c + dc))

        return min_r, max_r, min_c, max_c, region_cells

    # Find all connected components
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                min_r, max_r, min_c, max_c, region_cells = get_object_bounds(r, c)

                height = max_r - min_r + 1
                width = max_c - min_c + 1

                # Extract the object
                obj = []
                for i in range(height):
                    row = []
                    for j in range(width):
                        row.append(grid[min_r + i][min_c + j])
                    obj.append(row)

                # Score based on size, density, and pattern quality
                area = height * width
                density = len(region_cells) / area

                # Bonus for rectangular objects with good structure
                structure_bonus = 1.0
                if height >= 4 and width >= 4:
                    structure_bonus = 1.5

                # Bonus for objects with key pattern elements (4s and 6s/7s)
                has_4s = any(4 in row for row in obj)
                has_6s_or_7s = any(6 in row or 7 in row for row in obj)
                if has_4s and has_6s_or_7s:
                    structure_bonus *= 1.3

                score = area * density * structure_bonus

                if (
                    area >= 12 and density > 0.15
                ):  # Minimum size and density requirements
                    objects.append((obj, min_r, min_c, score, height, width))

    if len(objects) < 2:
        return [[0]]

    # Sort by score and find the two best non-overlapping objects
    objects.sort(key=lambda x: x[3], reverse=True)

    obj1 = obj2 = None
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            o1, r1, c1, s1, h1, w1 = objects[i]
            o2, r2, c2, s2, h2, w2 = objects[j]

            # Check overlap
            overlap_r = max(0, min(r1 + h1, r2 + h2) - max(r1, r2))
            overlap_c = max(0, min(c1 + w1, c2 + w2) - max(c1, c2))
            overlap_area = overlap_r * overlap_c

            min_area = min(h1 * w1, h2 * w2)
            if overlap_area < 0.3 * min_area:  # Less than 30% overlap
                obj1, obj2 = o1, o2
                break

        if obj1 and obj2:
            break

    if not obj1 or not obj2:
        # Use top two objects regardless of overlap
        obj1 = objects[0][0]
        obj2 = objects[1][0] if len(objects) > 1 else obj1

    # Make objects same size by trimming to minimum dimensions
    h1, w1 = len(obj1), len(obj1[0])
    h2, w2 = len(obj2), len(obj2[0])
    min_h, min_w = min(h1, h2), min(w1, w2)

    obj1_trimmed = [row[:min_w] for row in obj1[:min_h]]
    obj2_trimmed = [row[:min_w] for row in obj2[:min_h]]

    # Compare objects and create difference map
    result = []
    for i in range(min_h):
        row = []
        has_diff = False
        for j in range(min_w):
            if obj1_trimmed[i][j] != obj2_trimmed[i][j]:
                row.append(8)
                has_diff = True
            else:
                row.append(0)

        # Only keep rows that have differences and contain meaningful content
        has_content = any(
            obj1_trimmed[i][j] != 0 or obj2_trimmed[i][j] != 0 for j in range(min_w)
        )
        if has_diff and has_content:
            result.append(row)

    return result if result else [[0]]
