def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a 5-frame containing solid-color blocks
    2. Top-left corner regions (4x4 blocks) define color pair swaps
    3. Top section contains pattern templates for each color
    4. Each solid block in frame is replaced with its pattern, swapping to pair color

    Procedure:
    1. Find 5-frame boundaries and extract top section and frame content
    2. Identify color pairs from corner regions (including frame boundary row)
    3. Extract pattern templates for each color from top section
    4. Use flood-fill to identify solid-color blocks in frame
    5. Transform each block by applying pattern with color swapping
    6. Wrap result with 5-border
    """

    # Find 5-frame boundaries
    frame_rows = set()
    frame_cols = set()
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 5:
                frame_rows.add(r)
                frame_cols.add(c)

    if not frame_rows:
        return grid

    fr_min = min(frame_rows)
    fr_max = max(frame_rows)
    fc_min = min(frame_cols)
    fc_max = max(frame_cols)

    # Extract sections - include row with frame start in search area
    top = [row[:] for row in grid[:fr_min + 1]]  # Include frame start row
    frame = []
    for r in range(fr_min + 1, fr_max):
        frame.append(grid[r][fc_min + 1:fc_max])

    # Find background (most common in top, excluding frame row)
    counts = {}
    for row in grid[:fr_min]:  # Count only above frame
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=counts.get) if counts else 1

    # Find color pairs from corner regions (check wider area)
    def get_colors(r1, r2, c1, c2):
        s = set()
        for r in range(r1, min(r2, len(top))):
            for c in range(c1, min(c2, len(top[0]))):
                if top[r][c] != bg and top[r][c] != 5:  # Exclude bg and frame
                    s.add(top[r][c])
        return s

    pairs = {}

    # Try multiple corner regions including near-frame area
    regions = [(0, 4, 0, 4), (4, 8, 0, 4), (0, 4, 4, 8), (4, 9, 0, 4), (fr_min - 1, fr_min + 1, 0, 4)]
    for (r1, r2, c1, c2) in regions:
        cols = get_colors(r1, r2, c1, c2)
        if len(cols) == 2:
            a, b = list(cols)
            if a not in pairs:
                pairs[a] = b
                pairs[b] = a

    # Extract patterns for each color
    def find_pattern(color):
        """Find pattern template for a color in top section"""
        best = None
        best_score = 0

        for r in range(len(top)):
            for c in range(len(top[0])):
                if top[r][c] != color:
                    continue

                for sz in [3, 4, 5]:
                    if r + sz > len(top) or c + sz > len(top[0]):
                        continue

                    pat = []
                    cnt = 0
                    valid = True

                    for dr in range(sz):
                        row_d = []
                        for dc in range(sz):
                            v = top[r + dr][c + dc]
                            row_d.append(v)
                            if v == color:
                                cnt += 1
                            elif v != bg:
                                valid = False
                                break
                        if not valid:
                            break
                        pat.append(row_d)

                    if valid and cnt > 0:
                        # Score: prefer away from corners and higher density
                        score = cnt + (10 if c >= 5 else 0)
                        if score > best_score:
                            best_score = score
                            best = pat

        return best

    pats = {}
    for color in pairs:
        p = find_pattern(color)
        if p:
            pats[color] = p

    if not pats:
        return grid

    # Analyze frame structure
    # Map each cell to which block it belongs to
    # First, identify all blocks
    visited = [[False] * len(frame[0]) for _ in range(len(frame))]
    blocks = []  # List of (row, col, height, width, color)

    for r in range(len(frame)):
        for c in range(len(frame[0])):
            if not visited[r][c] and frame[r][c] in pats:
                color = frame[r][c]
                # Flood fill to find block extent
                min_r, max_r = r, r
                min_c, max_c = c, c

                stack = [(r, c)]
                visited[r][c] = True

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < len(frame) and 0 <= nc < len(frame[0]):
                            if not visited[nr][nc] and frame[nr][nc] == color:
                                visited[nr][nc] = True
                                stack.append((nr, nc))
                                min_r = min(min_r, nr)
                                max_r = max(max_r, nr)
                                min_c = min(min_c, nc)
                                max_c = max(max_c, nc)

                h = max_r - min_r + 1
                w = max_c - min_c + 1
                blocks.append((min_r, min_c, h, w, color))

    # Build block lookup: map (r, c) → (block_r, block_c, color)
    block_map = {}
    for (br, bc, h, w, color) in blocks:
        for dr in range(h):
            for dc in range(w):
                block_map[(br + dr, bc + dc)] = (br, bc, color)

    # Transform frame
    result = []
    for r in range(len(frame)):
        row = []
        for c in range(len(frame[0])):
            if (r, c) in block_map:
                br, bc, color = block_map[(r, c)]
                # Position within block
                in_r = r - br
                in_c = c - bc

                # Apply pattern
                pat = pats[color]
                pat_sz = len(pat)

                if in_r < pat_sz and in_c < pat_sz:
                    pat_val = pat[in_r][in_c]
                    # Swap target color
                    if pat_val == color:
                        row.append(pairs[color])
                    else:
                        row.append(pat_val)
                else:
                    row.append(bg)
            else:
                row.append(frame[r][c])

        result.append(row)

    # Add border
    h = len(result)
    w = len(result[0]) if result else 0
    out = [[5] * (w + 2) for _ in range(h + 2)]
    for r in range(h):
        for c in range(w):
            out[r + 1][c + 1] = result[r][c]

    return out
