def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 30x30 grid with two rectangular frames
    2. First frame contains patterns (marked by special values at center)
    3. Second frame has markers (special values) indicating where to place patterns
    4. Output is the second frame with patterns from first frame overlaid at marker positions

    Procedure:
    1. Find both rectangular frames
    2. Extract patterns from first frame (identify by special center values)
    3. Find markers in second frame (non-background values)
    4. Match each marker to closest pattern with same center value
    5. Overlay patterns onto second frame at marker positions
    6. Return second frame with border
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    def find_frames():
        frames = []
        for r in range(rows - 2):
            for c in range(cols - 2):
                border_val = grid[r][c]

                # Try to find a frame starting at (r, c)
                for height in range(4, min(35, rows - r + 1)):
                    for width in range(4, min(35, cols - c + 1)):
                        r2, c2 = r + height - 1, c + width - 1

                        if r2 >= rows or c2 >= cols:
                            continue

                        # Check if this forms a valid frame
                        is_frame = True
                        for i in range(c, c2 + 1):
                            if grid[r][i] != border_val or grid[r2][i] != border_val:
                                is_frame = False
                                break
                        if is_frame:
                            for i in range(r, r2 + 1):
                                if grid[i][c] != border_val or grid[i][c2] != border_val:
                                    is_frame = False
                                    break

                        if is_frame and r2 > r + 1 and c2 > c + 1:
                            # Get interior background
                            counter = Counter()
                            for rr in range(r + 1, r2):
                                for cc in range(c + 1, c2):
                                    counter[grid[rr][cc]] += 1

                            if counter:
                                bg = counter.most_common(1)[0][0]
                                # Check if not already found
                                overlap = False
                                for f in frames:
                                    if not (r2 < f['r1'] or r > f['r2'] or c2 < f['c1'] or c > f['c2']):
                                        overlap = True
                                        break

                                if not overlap:
                                    frames.append({
                                        'border': border_val,
                                        'r1': r, 'c1': c, 'r2': r2, 'c2': c2,
                                        'bg': bg
                                    })
                                    break
                    if len([f for f in frames if f['r1'] == r and f['c1'] == c]) > 0:
                        break

        return frames

    frames = find_frames()
    if len(frames) < 2:
        return [[0]]

    # Sort frames: first by top position, then by left position
    frames.sort(key=lambda f: (f['r1'], f['c1']))

    first_frame = frames[0]
    second_frame = frames[1]

    # Extract markers from second frame first
    sr1, sc1, sr2, sc2 = second_frame['r1'], second_frame['c1'], second_frame['r2'], second_frame['c2']
    sbg = second_frame['bg']

    markers = []  # [(interior_r, interior_c, value)]
    marker_values = set()
    for r in range(sr1 + 1, sr2):
        for c in range(sc1 + 1, sc2):
            val = grid[r][c]
            if val != sbg and val != second_frame['border']:
                interior_r = r - sr1 - 1
                interior_c = c - sc1 - 1
                markers.append((interior_r, interior_c, val))
                marker_values.add(val)

    # Extract patterns from first frame (only those with marker values as centers)
    fr1, fc1, fr2, fc2 = first_frame['r1'], first_frame['c1'], first_frame['r2'], first_frame['c2']
    fbg = first_frame['bg']

    patterns = {}  # {(interior_r, interior_c): pattern_dict}

    for r in range(fr1 + 1, fr2):
        for c in range(fc1 + 1, fc2):
            val = grid[r][c]
            # Only extract patterns centered at marker values
            if val in marker_values:
                interior_r = r - fr1 - 1
                interior_c = c - fc1 - 1

                # Extract larger region around this point (5x5 to catch full pattern)
                pattern_data = {}
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if fr1 < nr < fr2 and fc1 < nc < fc2:
                            cell_val = grid[nr][nc]
                            # Only include non-background values in the pattern
                            if cell_val != fbg:
                                pattern_data[(dr, dc)] = cell_val

                patterns[(interior_r, interior_c)] = {
                    'center_val': val,
                    'data': pattern_data
                }

    # Create output based on second frame
    interior_h = sr2 - sr1 - 1
    interior_w = sc2 - sc1 - 1

    # Initialize with second frame's interior
    interior = []
    for r in range(sr1 + 1, sr2):
        row = []
        for c in range(sc1 + 1, sc2):
            row.append(grid[r][c])
        interior.append(row)

    # Match markers to patterns and overlay
    for marker_r, marker_c, marker_val in markers:
        # Find closest pattern with matching center value
        best_pattern = None
        best_dist = float('inf')

        for (pr, pc), pdata in patterns.items():
            if pdata['center_val'] == marker_val:
                dist = ((marker_r - pr) ** 2 + (marker_c - pc) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pattern = pdata

        # If no exact match, try any pattern
        if best_pattern is None:
            for (pr, pc), pdata in patterns.items():
                dist = ((marker_r - pr) ** 2 + (marker_c - pc) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pattern = pdata

        # Overlay pattern at marker position
        if best_pattern:
            for (dr, dc), val in best_pattern['data'].items():
                nr = marker_r + dr
                nc = marker_c + dc
                if 0 <= nr < interior_h and 0 <= nc < interior_w:
                    interior[nr][nc] = val

    # Build output with border
    result = []
    border_row = [second_frame['border']] * (interior_w + 2)
    result.append(border_row)

    for row in interior:
        result.append([second_frame['border']] + row + [second_frame['border']])

    result.append(border_row)

    return result
