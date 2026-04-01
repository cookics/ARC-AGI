def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has marker (0) indicating orientation
    2. Content divided by separator columns (all 8s) into strips
    3. Blocks are strip × row group combinations
    4. Blocks with height > 2 are transposed after reversing rows
    5. Blocks trimmed to bounding box and centered
    6. Sorting order depends on whether blocks align horizontally

    Procedure:
    1. Remove marker row/column
    2. Find separators and extract blocks per strip
    3. Transform tall blocks
    4. Sort and output
    """

    if not grid or not grid[0]:
        return []

    # Find and remove marker
    marker_row, marker_col = -1, -1
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                marker_row, marker_col = i, j
                break
        if marker_row != -1:
            break

    content = []
    for i in range(len(grid)):
        if i == marker_row:
            continue
        row = []
        for j in range(len(grid[i])):
            if j == marker_col:
                continue
            row.append(grid[i][j])
        if row:
            content.append(row)

    if not content:
        return []

    rows, cols = len(content), len(content[0])

    def trim(blk):
        """Trim to bounding box of non-8 values"""
        if not blk or not blk[0]:
            return blk
        rmin, rmax = len(blk), -1
        cmin, cmax = len(blk[0]), -1
        for i, row in enumerate(blk):
            for j, val in enumerate(row):
                if val != 8:
                    rmin = min(rmin, i)
                    rmax = max(rmax, i)
                    cmin = min(cmin, j)
                    cmax = max(cmax, j)
        if rmax == -1:
            return blk
        return [blk[i][cmin:cmax+1] for i in range(rmin, rmax+1)]

    # Find separator columns
    seps = []
    for col in range(cols):
        if all(content[r][col] == 8 for r in range(rows)):
            seps.append(col)

    # Get strips
    strips = []
    for i in range(len(seps) - 1):
        if seps[i] + 1 < seps[i + 1]:
            strips.append((seps[i] + 1, seps[i + 1]))

    # Extract blocks - split by internal separators within strips
    blocks = []
    for sc, ec in strips:
        r_start = -1
        for r in range(rows):
            has_val = any(content[r][c] != 8 for c in range(sc, ec))
            if has_val and r_start == -1:
                r_start = r
            elif not has_val and r_start != -1:
                # Extract block
                blk = [content[rr][sc:ec] for rr in range(r_start, r)]
                # Find internal separators and split
                internal_seps = []
                for c in range(len(blk[0])):
                    if all(blk[rr][c] == 8 for rr in range(len(blk))):
                        internal_seps.append(c)

                # Split into sub-blocks
                prev = -1
                for sep in internal_seps + [len(blk[0])]:
                    if prev + 1 < sep:
                        sub_blk = [row[prev+1:sep] for row in blk]
                        sub_blk = trim(sub_blk)
                        if sub_blk:
                            blocks.append((sc + prev + 1, r_start, sub_blk))
                    prev = sep
                r_start = -1
        if r_start != -1:
            blk = [content[rr][sc:ec] for rr in range(r_start, rows)]
            # Find internal separators and split
            internal_seps = []
            for c in range(len(blk[0])):
                if all(blk[rr][c] == 8 for rr in range(len(blk))):
                    internal_seps.append(c)

            # Split into sub-blocks
            prev = -1
            for sep in internal_seps + [len(blk[0])]:
                if prev + 1 < sep:
                    sub_blk = [row[prev+1:sep] for row in blk]
                    sub_blk = trim(sub_blk)
                    if sub_blk:
                        blocks.append((sc + prev + 1, r_start, sub_blk))
                prev = sep

    # Transform
    trans = []
    for sc, rs, blk in blocks:
        h = len(blk)
        if h > 2:
            # Reverse each row first
            blk = [row[::-1] for row in blk]
            # Then transpose
            blk = [[blk[r][c] for r in range(h)] for c in range(len(blk[0]))]
        trans.append((sc, rs, blk))

    # Sort
    # Determine sorting based on number of blocks per row and marker position
    row_counts = {}
    for _, rs, _ in trans:
        row_counts[rs] = row_counts.get(rs, 0) + 1

    # Check if marker is on the right (affects left-to-right vs right-to-left order)
    marker_on_right = marker_col == len(grid[0]) - 1

    # If all rows have exactly 1 block each OR only 2 strips with same width -> sort by (col, row)
    # Otherwise sort by (row, col) or (row, -col) depending on marker position
    strip_widths = [ec - sc for sc, ec in strips]
    all_single = all(cnt == 1 for cnt in row_counts.values())
    same_width_2strips = len(strips) == 2 and len(set(strip_widths)) == 1

    if all_single or same_width_2strips:
        if marker_on_right:
            trans.sort(key=lambda x: (-x[0], x[1]))
        else:
            trans.sort(key=lambda x: (x[0], x[1]))
    else:
        if marker_on_right:
            trans.sort(key=lambda x: (x[1], -x[0]))
        else:
            trans.sort(key=lambda x: (x[1], x[0]))

    # Output
    max_w = max(len(blk[0]) for _, _, blk in trans) if trans else 0
    result = []
    for _, _, blk in trans:
        w = len(blk[0])
        pl = (max_w - w) // 2
        pr = max_w - w - pl
        for row in blk:
            result.append([8] * pl + row + [8] * pr)

    return result
