def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid with background, separator lines (vertical or horizontal)
    2. Separators have colored segments marking positions
    3. Two isolated markers of rare color
    4. Connect markers through segments with rectangular frames
    5. Horizontal seps: L-paths through segment centers
    6. Vertical seps: Rectangular frames wrapping segments

    Procedure:
    1. Find markers and separator type
    2. Find all segments on separators
    3. Draw connecting frames from marker1 through segments to marker2
    4. Each segment gets a frame extending to adjacent elements
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find markers
    value_counts = Counter(val for row in grid for val in row)
    rare_vals = [v for v, cnt in value_counts.items() if cnt <= 3]

    if not rare_vals:
        return result

    marker_val = rare_vals[0]
    markers = [(r, c) for r in range(rows) for c in range(cols)
               if grid[r][c] == marker_val]

    if len(markers) != 2:
        return result

    markers.sort()

    # Detect separators and segments
    h_seps = []
    for r in range(rows):
        counts = Counter(grid[r])
        if len(counts) >= 2:
            sep_val, sep_count = counts.most_common(1)[0]
            if sep_count > cols * 0.8:
                for c in range(cols):
                    if grid[r][c] != sep_val:
                        start = c
                        while c < cols and grid[r][c] != sep_val:
                            c += 1
                        center = (start + c - 1) // 2
                        h_seps.append((r, center))
                        break

    v_seps = []
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        counts = Counter(col_vals)
        if len(counts) >= 2:
            sep_val, sep_count = counts.most_common(1)[0]
            if sep_count > rows * 0.7:
                r = 0
                segs = []
                while r < rows:
                    if grid[r][c] != sep_val:
                        start = r
                        while r < rows and grid[r][c] != sep_val:
                            r += 1
                        segs.append((start, r - 1))
                    else:
                        r += 1
                if segs:
                    v_seps.append((c, segs))

    if len(h_seps) > len(v_seps):
        # Horizontal separators
        curr_r, curr_c = markers[0]

        for sep_r, seg_c in h_seps:
            for r in range(curr_r + 1, sep_r):
                result[r][curr_c] = marker_val
            for c in range(min(curr_c, seg_c), max(curr_c, seg_c) + 1):
                result[sep_r - 1][c] = marker_val
            result[sep_r][seg_c] = marker_val
            curr_r, curr_c = sep_r, seg_c

        end_r, end_c = markers[1]
        for r in range(curr_r + 1, end_r):
            result[r][curr_c] = marker_val
        for c in range(min(curr_c, end_c), max(curr_c, end_c) + 1):
            result[end_r][c] = marker_val

    else:
        # Vertical separators
        all_segs = [(c, start, end) for c, segs in v_seps for start, end in segs]
        all_segs.sort()

        if len(all_segs) == 0:
            return result

        # For each segment, draw frame
        for i, (seg_c, seg_start, seg_end) in enumerate(all_segs):
            if i == 0:
                # First segment - connects to first marker
                m_r, m_c = markers[0]

                # Frame from segment to marker
                frame_top = seg_start + 1
                frame_bottom = m_r
                frame_left = seg_c - 1
                frame_right = all_segs[i+1][0] - 1 if i+1 < len(all_segs) else seg_c + 1

                # Top horizontal
                if 0 <= frame_top < rows:
                    for c in range(max(0, frame_left), min(cols, frame_right + 1)):
                        if c != seg_c:
                            result[frame_top][c] = marker_val

                # Left vertical (from top to bottom)
                if 0 <= frame_left < cols:
                    for r in range(max(0, frame_top), min(rows, frame_bottom + 1)):
                        result[r][frame_left] = marker_val

                # Right vertical
                if frame_right != seg_c and 0 <= frame_right < cols:
                    for r in range(max(0, frame_top), min(rows, frame_bottom + 1)):
                        result[r][frame_right] = marker_val

                # Connect marker to frame left
                if 0 <= m_r < rows:
                    for c in range(max(0, min(m_c, frame_left)), min(cols, max(m_c, frame_left) + 1)):
                        result[m_r][c] = marker_val

            elif i == len(all_segs) - 1:
                # Last segment - connects to second marker
                m_r, m_c = markers[1]

                frame_top = seg_start + 1
                frame_bottom = m_r
                frame_left = all_segs[i-1][0] + 1
                frame_right = seg_c + 1

                # Top horizontal
                if 0 <= frame_top < rows:
                    for c in range(max(0, frame_left), min(cols, frame_right + 1)):
                        if c != seg_c:
                            result[frame_top][c] = marker_val

                # Left vertical
                if frame_left != seg_c and 0 <= frame_left < cols:
                    for r in range(max(0, frame_top), min(rows, frame_bottom + 1)):
                        result[r][frame_left] = marker_val

                # Right vertical
                if 0 <= frame_right < cols:
                    for r in range(max(0, frame_top), min(rows, frame_bottom + 1)):
                        result[r][frame_right] = marker_val

                # Connect marker to frame right
                if 0 <= m_r < rows:
                    for c in range(max(0, min(m_c, frame_right)), min(cols, max(m_c, frame_right) + 1)):
                        result[m_r][c] = marker_val

            else:
                # Middle segment - bridges two frames
                frame_top = seg_start
                frame_bottom = seg_end - 1
                frame_left = all_segs[i-1][0] + 1
                frame_right = all_segs[i+1][0] - 1

                # Top and bottom horizontals
                if 0 <= frame_top < rows:
                    for c in range(max(0, frame_left), min(cols, frame_right + 1)):
                        if c != seg_c:
                            result[frame_top][c] = marker_val
                if 0 <= frame_bottom < rows:
                    for c in range(max(0, frame_left), min(cols, frame_right + 1)):
                        if c != seg_c:
                            result[frame_bottom][c] = marker_val

                # Left and right verticals
                for r in range(max(0, frame_top), min(rows, frame_bottom + 1)):
                    if frame_left != seg_c and 0 <= frame_left < cols:
                        result[r][frame_left] = marker_val
                    if frame_right != seg_c and 0 <= frame_right < cols:
                        result[r][frame_right] = marker_val

    return result
