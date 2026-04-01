def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has background, pattern structures (1/4/8), and markers (2)
    2. Patterns form rectangular frames with hollow interiors
    3. Groups of 2s move INTO pattern interiors along aligned rows/columns
    4. For each frame, find the closest aligned group and pull it in
    5. Path fills with 0s from source to destination (excluding destination)
    6. Unmatched 2s removed

    Procedure:
    1. Find background and pattern values
    2. Find patterns as connected components
    3. For each pattern, identify interior cells and available rows/cols
    4. Group all 2s into connected rectangular regions
    5. For each pattern, find best aligned 2-group (by distance)
    6. Move group into interior, fill path with 0s
    7. Remove unmatched 2s
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    from collections import Counter
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_values).most_common(1)[0][0]

    # Find all 2s and group them
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]
    if not twos:
        return result

    visited = set()
    groups = []

    for r, c in twos:
        if (r, c) in visited:
            continue

        # BFS to find connected 2s
        queue = [(r, c)]
        group = set()
        visited.add((r, c))
        min_r, max_r = r, r
        min_c, max_c = c, c

        while queue:
            cr, cc = queue.pop(0)
            group.add((cr, cc))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 2:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    min_r, max_r = min(min_r, nr), max(max_r, nr)
                    min_c, max_c = min(min_c, nc), max(max_c, nc)

        groups.append((min_r, max_r, min_c, max_c, group))

    # Find pattern value
    pattern_value = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and grid[r][c] != 2:
                pattern_value = grid[r][c]
                break
        if pattern_value:
            break

    if not pattern_value:
        for r in range(rows):
            for c in range(cols):
                if result[r][c] == 2:
                    result[r][c] = background
        return result

    # Find all pattern structures
    pattern_visited = set()
    patterns = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == pattern_value and (r, c) not in pattern_visited:
                queue = [(r, c)]
                pattern_cells = set()
                pattern_visited.add((r, c))
                min_r, max_r = r, r
                min_c, max_c = c, c

                while queue:
                    cr, cc = queue.pop(0)
                    pattern_cells.add((cr, cc))

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in pattern_visited and grid[nr][nc] == pattern_value:
                            pattern_visited.add((nr, nc))
                            queue.append((nr, nc))
                            min_r, max_r = min(min_r, nr), max(max_r, nr)
                            min_c, max_c = min(min_c, nc), max(max_c, nc)

                # Compute interior cells and available rows/cols
                interior = set()
                interior_rows = set()
                interior_cols = set()

                for ir in range(min_r, max_r + 1):
                    for ic in range(min_c, max_c + 1):
                        if (ir, ic) not in pattern_cells:
                            interior.add((ir, ic))
                            interior_rows.add(ir)
                            interior_cols.add(ic)

                if interior:
                    patterns.append({
                        'bbox': (min_r, max_r, min_c, max_c),
                        'cells': pattern_cells,
                        'interior': interior,
                        'interior_rows': interior_rows,
                        'interior_cols': interior_cols
                    })

    # Match groups to patterns
    matched = set()

    for pattern in patterns:
        min_rp, max_rp, min_cp, max_cp = pattern['bbox']
        interior_rows = pattern['interior_rows']
        interior_cols = pattern['interior_cols']
        interior = pattern['interior']

        # Find best matching group for this pattern
        best_match = None
        best_distance = float('inf')

        for min_r2, max_r2, min_c2, max_c2, group in groups:
            if group & matched:  # Skip if already matched
                continue

            group_height = max_r2 - min_r2 + 1
            group_width = max_c2 - min_c2 + 1
            group_cols = set(range(min_c2, max_c2 + 1))
            group_rows = set(range(min_r2, max_r2 + 1))

            # Check vertical movement (group above/below pattern, columns align)
            if (group_cols.issubset(interior_cols) and
                min_c2 >= min_cp and max_c2 <= max_cp):
                if max_r2 < min_rp:  # Group above, move down into frame
                    # Check if top edge is open at group columns
                    if all((min_rp, c) not in pattern['cells'] for c in group_cols):
                        target_r = None
                        for r in sorted(interior_rows):
                            if all(all((r + offset, c) in interior for c in group_cols) for offset in range(group_height)):
                                target_r = r
                                break

                        if target_r is not None:
                            distance = min_rp - max_r2
                            if distance < best_distance:
                                best_distance = distance
                                best_match = ('v_down', min_r2, max_r2, min_c2, max_c2, group, target_r)

                elif min_r2 > max_rp:  # Group below, move up into frame
                    # Check if bottom edge is open at group columns
                    if all((max_rp, c) not in pattern['cells'] for c in group_cols):
                        target_r = None
                        for r in sorted(interior_rows):
                            if all(all((r + offset, c) in interior for c in group_cols) for offset in range(group_height)):
                                target_r = r
                                break

                        if target_r is not None:
                            distance = min_r2 - max_rp
                            if distance < best_distance:
                                best_distance = distance
                                best_match = ('v_up', min_r2, max_r2, min_c2, max_c2, group, target_r)

            # Check horizontal movement (group left/right of pattern, rows align)
            # Groups must not be at boundary rows
            if (group_rows.issubset(interior_rows) and
                min_r2 >= min_rp and max_r2 <= max_rp and
                min_r2 != min_rp and max_r2 != max_rp):
                if max_c2 < min_cp:  # Group left, move right into frame
                    # Check if left edge is open at group rows
                    if all((r, min_cp) not in pattern['cells'] for r in group_rows):
                        target_c = None
                        for c in sorted(interior_cols):
                            # Target must not be at boundary column
                            if c != min_cp and c != max_cp:
                                if all(all((r, c + offset) in interior for r in group_rows) for offset in range(group_width)):
                                    target_c = c
                                    break

                        if target_c is not None:
                            distance = min_cp - max_c2
                            if distance < best_distance:
                                best_distance = distance
                                best_match = ('h_right', min_r2, max_r2, min_c2, max_c2, group, target_c)

                elif min_c2 > max_cp:  # Group right, move left into frame
                    # Check if right edge is open at group rows
                    if all((r, max_cp) not in pattern['cells'] for r in group_rows):
                        target_c = None
                        for c in sorted(interior_cols):
                            # Target must not be at boundary column
                            if c != min_cp and c != max_cp:
                                if all(all((r, c + offset) in interior for r in group_rows) for offset in range(group_width)):
                                    target_c = c
                                    break

                        if target_c is not None:
                            distance = min_c2 - max_cp
                            if distance < best_distance:
                                best_distance = distance
                                best_match = ('h_left', min_r2, max_r2, min_c2, max_c2, group, target_c)

        # Apply best match if found
        if best_match:
            direction = best_match[0]
            min_r2, max_r2, min_c2, max_c2, group = best_match[1:6]
            target = best_match[6]
            group_cols = set(range(min_c2, max_c2 + 1))
            group_rows = set(range(min_r2, max_r2 + 1))

            if direction == 'v_down':
                # Place 2s at target
                for r, c in group:
                    offset_r = r - min_r2
                    result[target + offset_r][c] = 2
                # Fill path from original to target (not including target)
                for r in range(min_r2, target):
                    for c in group_cols:
                        result[r][c] = 0

            elif direction == 'v_up':
                # Place 2s at target
                for r, c in group:
                    offset_r = r - min_r2
                    result[target + offset_r][c] = 2
                # Fill path from target end to original (including original)
                group_height = max_r2 - min_r2 + 1
                for r in range(target + group_height, max_r2 + 1):
                    for c in group_cols:
                        result[r][c] = 0

            elif direction == 'h_right':
                # Place 2s at target
                for r, c in group:
                    offset_c = c - min_c2
                    result[r][target + offset_c] = 2
                # Fill path from original to target (not including target)
                for r in group_rows:
                    for c in range(min_c2, target):
                        result[r][c] = 0

            elif direction == 'h_left':
                # Place 2s at target
                for r, c in group:
                    offset_c = c - min_c2
                    result[r][target + offset_c] = 2
                # Fill path from target end to original (including original)
                group_width = max_c2 - min_c2 + 1
                for r in group_rows:
                    for c in range(target + group_width, max_c2 + 1):
                        result[r][c] = 0

            matched.update(group)

    # Remove unmatched 2s
    for r, c in twos:
        if (r, c) not in matched:
            result[r][c] = background

    return result
