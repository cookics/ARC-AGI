def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has repeating background (0,1,2,3...)
    2. Largest hollow frame becomes output border
    3. Other frames extracted WITH all nested content (preserves nesting)
    4. Each frame+content region placed in output at specific position

    Procedure:
    1. Find background values
    2. Find all hollow frames
    3. Select largest frame as main border
    4. For each secondary frame, extract entire region (frame + interior)
    5. Place regions in output, preserving their vertical positions
    """

    rows, cols = len(grid), len(grid[0])
    from collections import Counter

    # Identify background
    all_vals = [grid[r][c] for r in range(rows) for c in range(cols)]
    counts = Counter(all_vals)

    # Find hollow frames first
    def find_frame(val):
        ps = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == val]
        if len(ps) < 12:
            return None
        r1, r2 = min(p[0] for p in ps), max(p[0] for p in ps)
        c1, c2 = min(p[1] for p in ps), max(p[1] for p in ps)
        h, w = r2 - r1 + 1, c2 - c1 + 1
        if h < 3 or w < 3:
            return None
        border = {(r, c) for r in [r1, r2] for c in range(c1, c2 + 1)}
        border |= {(r, c) for c in [c1, c2] for r in range(r1, r2 + 1)}
        if len(set(ps) & border) / len(border) > 0.6:
            return {'val': val, 'r1': r1, 'r2': r2, 'c1': c1, 'c2': c2,
                    'h': h, 'w': w, 'area': h * w}
        return None

    # Find all frames
    all_frames = []
    for v in counts:
        f = find_frame(v)
        if f:
            all_frames.append(f)

    # Determine background: 0,1,2 are always background
    # 3 is background UNLESS it forms a frame
    bg = {0, 1, 2}
    if 3 not in [f['val'] for f in all_frames]:
        bg.add(3)

    frames = [f for f in all_frames if f['val'] not in bg]
    if not frames:
        return [[0]]

    # Find main frame (largest)
    main = max(frames, key=lambda x: x['area'])
    fr1, fr2, fc1, fc2 = main['r1'], main['r2'], main['c1'], main['c2']
    oh, ow = main['h'], main['w']

    # Create output with main frame border
    result = [[0] * ow for _ in range(oh)]
    for c in range(ow):
        result[0][c] = result[oh - 1][c] = main['val']
    for r in range(oh):
        result[r][0] = result[r][ow - 1] = main['val']

    # Process secondary frames (extract with all nested content)
    secondary_frames = [f for f in frames if f['val'] != main['val']]

    for frame in secondary_frames:
        vr1, vr2, vc1, vc2 = frame['r1'], frame['r2'], frame['c1'], frame['c2']
        vh, vw = frame['h'], frame['w']
        frame_val = frame['val']

        # Extract frame with its nested content
        region = [[0] * vw for _ in range(vh)]
        for dr in range(vh):
            for dc in range(vw):
                src_r, src_c = vr1 + dr, vc1 + dc
                if 0 <= src_r < rows and 0 <= src_c < cols:
                    val = grid[src_r][src_c]
                    # Include the frame color itself
                    if val == frame_val:
                        region[dr][dc] = val
                    # Include non-background values in the frame interior
                    elif val not in bg and val != main['val']:
                        # Check if this cell is in the interior of the frame
                        is_interior = 0 < dr < vh - 1 and 0 < dc < vw - 1
                        if is_interior:
                            region[dr][dc] = val

        # Check if this shape is inside the main frame (column-wise)
        inside_frame = vc1 >= fc1 and vc2 <= fc2

        # Position calculation
        if inside_frame:
            # Inside frame: use relative positioning
            target_r = vr1 - fr1
        else:
            # Outside frame: use absolute positioning
            target_r = vr1

        target_c = 1

        # Adjust if needed
        if target_r < 1:
            target_r = 1
        if target_r + vh > oh - 1:
            target_r = oh - vh - 1
        if target_c + vw > ow - 1:
            target_c = ow - vw - 1

        # Place region (overlay)
        for dr in range(vh):
            for dc in range(vw):
                if region[dr][dc] != 0:
                    out_r, out_c = target_r + dr, target_c + dc
                    if 1 <= out_r < oh - 1 and 1 <= out_c < ow - 1:
                        result[out_r][out_c] = region[dr][dc]

    # Process standalone blocks (not frames)
    for val in counts:
        if val in bg or val == main['val']:
            continue
        # Skip if this value is a frame
        if any(f['val'] == val for f in secondary_frames):
            continue

        # Extract blocks of this color
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == val]
        if not cells:
            continue

        vr1, vr2 = min(c[0] for c in cells), max(c[0] for c in cells)
        vc1, vc2 = min(c[1] for c in cells), max(c[1] for c in cells)
        vh, vw = vr2 - vr1 + 1, vc2 - vc1 + 1

        # Extract region
        region = [[0] * vw for _ in range(vh)]
        for r, c in cells:
            region[r - vr1][c - vc1] = val

        # Check if inside the main frame
        inside_frame = vc1 >= fc1 and vc2 <= fc2

        # Position calculation
        if inside_frame:
            target_r = vr1 - fr1
        else:
            target_r = vr1

        target_c = 1

        if target_r < 1:
            target_r = 1
        if target_r + vh > oh - 1:
            target_r = oh - vh - 1
        if target_c + vw > ow - 1:
            target_c = ow - vw - 1

        # Place (overlay)
        for dr in range(vh):
            for dc in range(vw):
                if region[dr][dc] != 0:
                    out_r, out_c = target_r + dr, target_c + dc
                    if 1 <= out_r < oh - 1 and 1 <= out_c < ow - 1:
                        result[out_r][out_c] = region[dr][dc]

    return result
