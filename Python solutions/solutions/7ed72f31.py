def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains colored patterns and lines/points of value 2
    2. Patterns reflect across their associated 2s lines
    3. Vertical lines reflect patterns horizontally (same row range)
    4. Horizontal lines reflect patterns vertically (same column range)
    5. Isolated 2s do point reflections

    Procedure:
    1. Find background color (most common)
    2. Find all 2s segments (horizontal lines, vertical lines, or isolated points)
    3. For each non-background, non-2 cell, find best matching segment
    4. Reflect cell across the segment
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Find background color
    flat = [cell for row in grid for cell in row]
    background = Counter(flat).most_common(1)[0][0]

    # Find all segments of 2s
    segments = []
    visited_twos = set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and (r, c) not in visited_twos:
                # Check for horizontal segment
                h_segment = [(r, c)]
                for c2 in range(c + 1, cols):
                    if grid[r][c2] == 2:
                        h_segment.append((r, c2))
                    else:
                        break
                for c2 in range(c - 1, -1, -1):
                    if grid[r][c2] == 2:
                        h_segment.insert(0, (r, c2))
                    else:
                        break

                # Check for vertical segment
                v_segment = [(r, c)]
                for r2 in range(r + 1, rows):
                    if grid[r2][c] == 2:
                        v_segment.append((r2, c))
                    else:
                        break
                for r2 in range(r - 1, -1, -1):
                    if grid[r2][c] == 2:
                        v_segment.insert(0, (r2, c))
                    else:
                        break

                # Determine segment type (2+ consecutive 2s form a line)
                if len(h_segment) >= 2 and len(v_segment) == 1:
                    segments.append(('horizontal', h_segment))
                    for pos in h_segment:
                        visited_twos.add(pos)
                elif len(v_segment) >= 2 and len(h_segment) == 1:
                    segments.append(('vertical', v_segment))
                    for pos in v_segment:
                        visited_twos.add(pos)
                elif len(h_segment) == 1 and len(v_segment) == 1:
                    segments.append(('point', [(r, c)]))
                    visited_twos.add((r, c))
                else:
                    # Cross or T-shape (both >= 2), treat as point for each 2
                    segments.append(('point', [(r, c)]))
                    visited_twos.add((r, c))

    # Reflect cells
    result = [row[:] for row in grid]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and grid[r][c] != 2:
                color = grid[r][c]

                # Find best matching segment
                best_score = float('inf')
                best_segment = None

                for seg_type, seg_cells in segments:
                    if seg_type == 'horizontal':
                        axis_r = seg_cells[0][0]
                        min_c = min(sc for sr, sc in seg_cells)
                        max_c = max(sc for sr, sc in seg_cells)

                        # Check if c is within segment's column range
                        if min_c <= c <= max_c:
                            row_dist = abs(r - axis_r)
                            if row_dist < best_score:
                                best_score = row_dist
                                best_segment = (seg_type, seg_cells)

                    elif seg_type == 'vertical':
                        axis_c = seg_cells[0][1]
                        min_r = min(sr for sr, sc in seg_cells)
                        max_r = max(sr for sr, sc in seg_cells)

                        # Check if r is within segment's row range
                        if min_r <= r <= max_r:
                            col_dist = abs(c - axis_c)
                            if col_dist < best_score:
                                best_score = col_dist
                                best_segment = (seg_type, seg_cells)

                    elif seg_type == 'point':
                        axis_r, axis_c = seg_cells[0]
                        dist = max(abs(r - axis_r), abs(c - axis_c))
                        if dist < best_score:
                            best_score = dist
                            best_segment = (seg_type, seg_cells)

                if best_segment is None:
                    continue

                seg_type, seg_cells = best_segment

                # Reflect across segment
                if seg_type == 'point':
                    axis_r, axis_c = seg_cells[0]
                    new_r = 2 * axis_r - r
                    new_c = 2 * axis_c - c
                    if 0 <= new_r < rows and 0 <= new_c < cols:
                        result[new_r][new_c] = color
                elif seg_type == 'horizontal':
                    axis_r = seg_cells[0][0]
                    new_r = 2 * axis_r - r
                    if 0 <= new_r < rows:
                        result[new_r][c] = color
                elif seg_type == 'vertical':
                    axis_c = seg_cells[0][1]
                    new_c = 2 * axis_c - c
                    if 0 <= new_c < cols:
                        result[r][new_c] = color

    return result
