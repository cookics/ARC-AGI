def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has separator rows (all same value, typically 6)
    2. Separator rows divide grid into sections
    3. First section's colors determine transformation behavior
    4. If first section has {0,1,6}: first/last sections transform (8→3)
       Else: first unchanged, last transforms (8→2)
    5. Middle section has pattern blocks to be framed

    Procedure:
    1. Detect separator rows
    2. Extract first section colors
    3. Apply section transforms
    4. Find and frame matching color patterns in middle section
    """
    import copy
    from collections import Counter

    result = copy.deepcopy(grid)
    H, W = len(grid), len(grid[0])

    # Find background value (most common)
    from collections import Counter as Cnt
    all_vals = [grid[r][c] for r in range(H) for c in range(W)]
    bg_val = Cnt(all_vals).most_common(1)[0][0]

    # Detect separator rows (rows where all values are same non-background value)
    separators = []
    for r in range(H):
        row_vals = set(grid[r])
        if len(row_vals) == 1 and bg_val not in row_vals:
            separators.append(r)

    if len(separators) < 2:
        return result

    # Define sections based on first two separators
    sep1, sep2 = separators[0], separators[1]
    sec1_end = sep1
    sec2_start = sep1 + 1
    sec2_end = sep2
    sec3_start = sep2 + 1

    def draw_frame(r0, r1, c0, c1):
        """Draw frame of 3s from (r0,c0) to (r1,c1), filling interior 8s"""
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r < sec2_start or r >= sec2_end:  # Out of Section 2
                    continue
                if len(set(grid[r])) == 1:  # Skip separator rows
                    continue
                if r == r0 or r == r1 or c == c0 or c == c1:  # Border
                    result[r][c] = 3
                elif result[r][c] == 8:  # Interior 8s
                    result[r][c] = 3

    # Get Section 1 colors (excluding background)
    s1_colors = set()
    s1_counts = Counter()
    sep_val = grid[sep1][0]  # Separator value
    for r in range(sec1_end):
        for c in range(W):
            if grid[r][c] != bg_val:
                s1_colors.add(grid[r][c])
                s1_counts[grid[r][c]] += 1

    # Find all unique non-background colors in entire grid
    all_colors = set(grid[r][c] for r in range(H) for c in range(W) if grid[r][c] != bg_val)

    # Get two smallest colors overall (potential base colors)
    sorted_colors = sorted(all_colors)
    base_color_1 = sorted_colors[0] if len(sorted_colors) >= 1 else None
    base_color_2 = sorted_colors[1] if len(sorted_colors) >= 2 else None

    # Transform if section 1 contains both smallest color values
    # (excluding separator) from the overall palette
    non_sep_s1_colors = s1_colors - {sep_val}
    has_transform = (base_color_1 in non_sep_s1_colors and
                    base_color_2 in non_sep_s1_colors)

    # Transform Section 1
    if has_transform:
        for r in range(sec1_end):
            for c in range(W):
                if result[r][c] == 8:
                    result[r][c] = 3

    # Transform Section 3
    s3_val = 3 if has_016 else 2
    for r in range(sec3_start, H):
        for c in range(W):
            if result[r][c] == 8:
                result[r][c] = s3_val

    # Find pattern blocks in Section 2 (groups of consecutive rows with non-8 content)
    blocks = []
    in_block = False
    block_start = None
    for r in range(sec2_start, sec2_end):
        has_content = any(grid[r][c] not in (6, 8) for c in range(W))
        if has_content and not in_block:
            block_start = r
            in_block = True
        elif not has_content and in_block:
            blocks.append((block_start, r - 1))
            in_block = False
    if in_block:
        blocks.append((block_start, sec2_end - 1))

    # Process each pattern block
    for r_start, r_end in blocks:
        # Find all colors in block and their bboxes
        color_info = {}
        for r in range(r_start, r_end + 1):
            for c in range(W):
                v = grid[r][c]
                if v not in (6, 8):
                    if v not in color_info:
                        # Compute bbox for this color
                        coords = [(rr, cc) for rr in range(r_start, r_end + 1)
                                 for cc in range(W) if grid[rr][cc] == v]
                        if coords:
                            rs = [x[0] for x in coords]
                            cs = [x[1] for x in coords]
                            color_info[v] = (min(rs), max(rs), min(cs), max(cs))

        # Get matching colors
        matches = {c: bb for c, bb in color_info.items() if c in s1_colors}
        if not matches:
            continue

        # Frame logic
        if has_016:
            # If all colors match, frame whole block
            if len(matches) == len(color_info) and len(matches) >= 4:
                # Frame full width
                min_c = min(bb[2] for bb in color_info.values())
                max_c = max(bb[3] for bb in color_info.values())
                draw_frame(r_start - 1, r_end + 1, max(1, min_c - 1), min(W - 2, max_c + 1))
            else:
                # Frame each match individually
                for c, (r0, r1, c0, c1) in matches.items():
                    draw_frame(max(sec2_start, r0 - 1), min(sec2_end - 1, r1 + 1),
                              max(1, c0 - 1), min(W - 2, c1 + 1))
        else:
            # Only frame most common color from Section 1
            if s1_counts:
                most_common = s1_counts.most_common(1)[0][0]
                regions = []
                for c, (r0, r1, c0, c1) in matches.items():
                    if len(matches) == 1 or c == most_common:
                        regions.append((max(sec2_start, r0 - 1), min(sec2_end - 1, r1 + 1),
                                       max(1, c0 - 1), min(W - 2, c1 + 1)))

                # Merge close regions
                if len(regions) > 1:
                    regions.sort(key=lambda x: x[2])
                    merged = [regions[0]]
                    for i in range(1, len(regions)):
                        if regions[i][2] - merged[-1][3] <= 2:
                            # Merge
                            merged[-1] = (min(merged[-1][0], regions[i][0]),
                                        max(merged[-1][1], regions[i][1]),
                                        min(merged[-1][2], regions[i][2]),
                                        max(merged[-1][3], regions[i][3]))
                        else:
                            merged.append(regions[i])
                    regions = merged

                for rr0, rr1, cc0, cc1 in regions:
                    draw_frame(rr0, rr1, cc0, cc1)

    return result
