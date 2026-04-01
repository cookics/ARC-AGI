def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has pattern regions with structural color (3/5/8) and embedded markers (1/2/4)
    2. Same marker values appear isolated elsewhere in the grid
    3. Pattern is transformed (rotated/flipped) and moved to align markers with isolated positions
    4. Original pattern is erased, new pattern appears at isolated marker locations

    Procedure:
    1. Find all connected components containing structural colors
    2. Identify which components are patterns (contain multiple marker values)
    3. Find isolated marker positions
    4. Match patterns to isolated marker groups by testing transformations
    5. Apply transformation to pattern and place at new location
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find connected components using flood fill
    visited = set()

    def flood_fill(r, c):
        if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == 0:
            return []
        visited.add((r, c))
        comp = [(r, c, grid[r][c])]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            comp.extend(flood_fill(r + dr, c + dc))
        return comp

    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                comp = flood_fill(r, c)
                if comp:
                    components.append(comp)

    # Identify structural color (most frequent among 3, 5, 8)
    struct_counts = {3: 0, 5: 0, 8: 0}
    for comp in components:
        for r, c, v in comp:
            if v in struct_counts:
                struct_counts[v] += 1

    struct_color = max(struct_counts, key=struct_counts.get)
    if struct_counts[struct_color] == 0:
        return [[0] * cols for _ in range(rows)]

    # Find pattern components (contain struct_color AND at least 2 different markers)
    marker_values = {1, 2, 4}
    patterns = []
    pattern_positions = set()

    for comp in components:
        has_struct = any(v == struct_color for r, c, v in comp)
        markers_in_comp = set(v for r, c, v in comp if v in marker_values)

        if has_struct and len(markers_in_comp) >= 2:
            patterns.append(comp)
            for r, c, v in comp:
                pattern_positions.add((r, c))

    # Find isolated markers (not in pattern components)
    isolated = [(r, c, grid[r][c]) for r in range(rows) for c in range(cols)
                if grid[r][c] in marker_values and (r, c) not in pattern_positions]

    if not patterns or not isolated:
        return [[0] * cols for _ in range(rows)]

    # Group isolated markers by proximity
    def manhattan_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    iso_groups = []
    used_iso = set()

    for i, marker in enumerate(isolated):
        if i in used_iso:
            continue
        group = [marker]
        used_iso.add(i)

        # Keep adding nearby markers
        changed = True
        while changed:
            changed = False
            for j, other in enumerate(isolated):
                if j not in used_iso:
                    if any(manhattan_dist(m[:2], other[:2]) <= 8 for m in group):
                        group.append(other)
                        used_iso.add(j)
                        changed = True

        iso_groups.append(group)

    # Try to match each pattern with each isolated group
    result = [[0] * cols for _ in range(rows)]

    def get_transformations():
        """Generate all 8 transformations (4 rotations × 2 flips)"""
        transforms = []
        # Identity, 90°, 180°, 270°
        transforms.append(lambda r, c: (r, c))
        transforms.append(lambda r, c: (c, -r))
        transforms.append(lambda r, c: (-r, -c))
        transforms.append(lambda r, c: (-c, r))
        # Horizontal flip + rotations
        transforms.append(lambda r, c: (r, -c))
        transforms.append(lambda r, c: (c, r))
        transforms.append(lambda r, c: (-r, c))
        transforms.append(lambda r, c: (-c, -r))
        return transforms

    for pattern in patterns:
        # Extract marker positions in pattern
        pattern_markers = [(r, c, v) for r, c, v in pattern if v in marker_values]

        if len(pattern_markers) < 2:
            continue

        # Try each isolated group
        for iso_group in iso_groups:
            # Check if marker values match
            pattern_marker_vals = set(v for r, c, v in pattern_markers)
            iso_marker_vals = set(v for r, c, v in iso_group)

            if pattern_marker_vals != iso_marker_vals:
                continue

            # Try all transformations
            for transform in get_transformations():
                # Use first marker as anchor
                p_anchor_r, p_anchor_c, p_anchor_v = pattern_markers[0]

                # Find corresponding isolated marker with same value
                iso_anchor = next((r, c, v) for r, c, v in iso_group if v == p_anchor_v)
                iso_anchor_r, iso_anchor_c, _ = iso_anchor

                # Check if all markers align
                all_match = True
                for p_r, p_c, p_v in pattern_markers:
                    # Transform pattern marker relative to anchor
                    dr, dc = transform(p_r - p_anchor_r, p_c - p_anchor_c)
                    new_r, new_c = iso_anchor_r + dr, iso_anchor_c + dc

                    # Check if there's an isolated marker at this position with same value
                    if not any(r == new_r and c == new_c and v == p_v for r, c, v in iso_group):
                        all_match = False
                        break

                if all_match:
                    # Apply transformation to entire pattern
                    for p_r, p_c, p_v in pattern:
                        dr, dc = transform(p_r - p_anchor_r, p_c - p_anchor_c)
                        new_r, new_c = iso_anchor_r + dr, iso_anchor_c + dc

                        if 0 <= new_r < rows and 0 <= new_c < cols:
                            result[new_r][new_c] = p_v

                    break

    return result
