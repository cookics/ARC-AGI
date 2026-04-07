def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains multiple hollow rectangles with uniform borders
    2. Output is the transformed interior of one or more frames
    3. Transformation rule:
       - First occurrence of rare values (like 6, 8) is preserved
       - Connected groups of common values become 2
       - Isolated common values become 0
       - 0 stays 0

    Procedure:
    1. Find all hollow rectangles with uniform borders
    2. For each frame, extract and transform its interior
    3. Combine transformed interiors if needed
    """

    rows, cols = len(grid), len(grid[0])

    def is_uniform_border(r1, c1, r2, c2, border_val):
        """Check if all border cells have the same value"""
        # Top and bottom edges
        for c in range(c1, c2 + 1):
            if grid[r1][c] != border_val or grid[r2][c] != border_val:
                return False
        # Left and right edges
        for r in range(r1, r2 + 1):
            if grid[r][c1] != border_val or grid[r][c2] != border_val:
                return False
        return True

    def extract_interior(r1, c1, r2, c2):
        """Extract interior cells (excluding border)"""
        interior = []
        for r in range(r1 + 1, r2):
            row = []
            for c in range(c1 + 1, c2):
                row.append(grid[r][c])
            interior.append(row)
        return interior

    # Find all rectangles
    rectangles = []

    for r1 in range(rows):
        for c1 in range(cols):
            border_val = grid[r1][c1]
            if border_val == 0:
                continue

            # Try all possible rectangle sizes
            max_height = rows - r1
            max_width = cols - c1

            for h in range(3, max_height + 1):
                for w in range(3, max_width + 1):
                    r2 = r1 + h - 1
                    c2 = c1 + w - 1

                    if is_uniform_border(r1, c1, r2, c2, border_val):
                        interior = extract_interior(r1, c1, r2, c2)
                        if interior and interior[0]:
                            interior_h = len(interior)
                            interior_w = len(interior[0])
                            rectangles.append({
                                'interior': interior,
                                'size': (interior_h, interior_w),
                                'border_val': border_val
                            })

    if not rectangles:
        return grid

    # Group by (border_val, size)
    from collections import defaultdict, Counter
    groups = defaultdict(list)

    for rect in rectangles:
        key = (rect['border_val'], rect['size'])
        groups[key].append(rect)

    # Find the group with most rectangles
    best_key = max(groups.keys(), key=lambda k: len(groups[k]))
    best_rects = groups[best_key]

    # Debug: print group information
    # print(f"Found {len(groups)} groups")
    # for key, rects in groups.items():
    #     print(f"  Group {key}: {len(rects)} rectangles")
    # print(f"Best group: {best_key} with {len(best_rects)} rectangles")

    if not best_rects:
        return grid

    # Transform each interior individually first
    def transform_interior(interior):
        h, w = len(interior), len(interior[0])
        result = [[0] * w for _ in range(h)]

        # Find all unique values
        from collections import Counter
        all_vals = []
        for row in interior:
            all_vals.extend(row)
        val_counts = Counter(all_vals)
        if 0 in val_counts:
            del val_counts[0]

        # Identify rare values (markers like 6, 8)
        if val_counts:
            sorted_vals = sorted(val_counts.items(), key=lambda x: x[1])
            # Rare values are those appearing less than 30% of most common
            max_count = max(val_counts.values())
            markers = {v for v, c in val_counts.items() if c < max_count * 0.3}
        else:
            markers = set()

        # Find connected components for each value
        def flood_fill(start_r, start_c, value, visited):
            if (start_r, start_c) in visited:
                return []
            if interior[start_r][start_c] != value:
                return []

            component = []
            stack = [(start_r, start_c)]
            while stack:
                r, c = stack.pop()
                if (r, c) in visited or r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if interior[r][c] != value:
                    continue
                visited.add((r, c))
                component.append((r, c))
                # 4-connectivity
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    stack.append((r + dr, c + dc))
            return component

        # Process each value
        visited = set()
        marker_seen = set()

        for r in range(h):
            for c in range(w):
                if (r, c) in visited or interior[r][c] == 0:
                    continue

                val = interior[r][c]
                component = flood_fill(r, c, val, visited)

                if val in markers:
                    # Marker value
                    if val not in marker_seen:
                        # First occurrence - preserve
                        for cr, cc in component:
                            result[cr][cc] = val
                        marker_seen.add(val)
                    else:
                        # Subsequent occurrence - convert to 2
                        for cr, cc in component:
                            result[cr][cc] = 2
                else:
                    # Common value
                    if len(component) == 1:
                        # Isolated cell - convert to 0
                        result[component[0][0]][component[0][1]] = 0
                    else:
                        # Connected group - convert to 2
                        for cr, cc in component:
                            result[cr][cc] = 2

        return result

    # Get transformed interiors
    transformed = [transform_interior(r['interior']) for r in best_rects]

    # If only one rectangle, return its transformed interior
    if len(transformed) == 1:
        return transformed[0]

    # Otherwise, combine by overlay (first non-zero wins, or use voting)
    h = len(transformed[0])
    w = len(transformed[0][0])
    result = [[0] * w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            values = [t[i][j] for t in transformed]
            # Use first non-zero value, or majority vote
            non_zero = [v for v in values if v != 0]
            if non_zero:
                counts = Counter(non_zero)
                result[i][j] = counts.most_common(1)[0][0]
            else:
                result[i][j] = 0

    return result
