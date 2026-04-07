def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Find 5-row patterns where top and bottom rows are identical
    2. Extract bounding boxes and combine horizontally
    3. Use matching sub-patterns for alignment

    Procedure:
    1. Find patterns with identical rows 4 apart or 2 apart
    2. Split each region into horizontal segments
    3. Combine segments with overlay based on suffix-prefix matching
    """

    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    value_counts = Counter(all_values)
    background = value_counts.most_common(1)[0][0]

    patterns = []
    used_rows = set()

    # Check if there are marker values (like 9)
    has_markers = 9 in value_counts

    # Find 5-row patterns
    for r1 in range(rows - 4):
        r2 = r1 + 4
        if r1 in used_rows:
            continue

        if grid[r1] == grid[r2]:
            min_col, max_col = cols, -1
            for r in range(r1, r2 + 1):
                for c in range(cols):
                    if grid[r][c] != background:
                        min_col = min(min_col, c)
                        max_col = max(max_col, c)

            if min_col <= max_col:
                # Don't split into segments - keep as one pattern
                region = [[grid[r][c] for c in range(min_col, max_col + 1)]
                         for r in range(r1, r2 + 1)]
                patterns.append((r1, min_col, region))
                used_rows.update(range(r1, r2 + 1))

    # Find 3-row patterns only if there are no marker values
    if not has_markers:
        for r1 in range(rows - 2):
            r2 = r1 + 2
            if r1 in used_rows:
                continue

            if grid[r1] == grid[r2]:
                start_row = max(0, r1 - 1)
                end_row = min(rows - 1, r2 + 1)

                while end_row - start_row < 4 and (start_row > 0 or end_row < rows - 1):
                    if start_row > 0:
                        start_row -= 1
                    elif end_row < rows - 1:
                        end_row += 1

                if end_row - start_row == 4:
                    min_col, max_col = cols, -1
                    for r in range(start_row, end_row + 1):
                        for c in range(cols):
                            if grid[r][c] != background:
                                min_col = min(min_col, c)
                                max_col = max(max_col, c)

                    if min_col <= max_col:
                        region = [[grid[r][c] for c in range(min_col, max_col + 1)]
                                 for r in range(start_row, end_row + 1)]
                        patterns.append((start_row, min_col, region))
                        used_rows.update(range(start_row, end_row + 1))

        # Also find column-wise matching segments for rows not yet used
        for r1 in range(rows - 4):
            r2 = r1 + 4
            if r1 in used_rows:
                continue

            # Find contiguous matching column segments
            matching_segs = []
            seg_start = None

            for c in range(cols):
                if grid[r1][c] == grid[r2][c]:
                    has_content = any(grid[r][c] != background for r in range(r1, r2 + 1))
                    if has_content:
                        if seg_start is None:
                            seg_start = c
                else:
                    if seg_start is not None:
                        matching_segs.append((seg_start, c - 1))
                        seg_start = None

            if seg_start is not None:
                matching_segs.append((seg_start, cols - 1))

            # Extract significant segments (at least 3 cols, substantial content)
            for seg_min, seg_max in matching_segs:
                if seg_max - seg_min >= 2:
                    content_count = sum(1 for r in range(r1, r2 + 1)
                                       for c in range(seg_min, seg_max + 1)
                                       if grid[r][c] != background)
                    if content_count >= 5:
                        region = [[grid[r][c] for c in range(seg_min, seg_max + 1)]
                                 for r in range(r1, r2 + 1)]
                        patterns.append((r1, seg_min, region))

    if not patterns:
        return []

    # Sort by column descending
    patterns.sort(key=lambda x: -x[1])

    # Combine with overlap
    def find_overlap(r1, r2, bg):
        best = 0
        for overlap_len in range(1, min(len(r1[0]), len(r2[0])) + 1):
            if all(r1[i][-overlap_len:] == r2[i][:overlap_len] for i in range(5)):
                if any(r1[i][-overlap_len] != bg for i in range(5)):
                    best = overlap_len
        return best

    result = [list(patterns[0][2][i]) for i in range(5)]

    for i in range(1, len(patterns)):
        _, _, next_region = patterns[i]
        overlap = find_overlap([result[j] for j in range(5)], next_region, background)

        for row_idx in range(5):
            if overlap > 0:
                for j in range(overlap):
                    idx = len(result[row_idx]) - overlap + j
                    if result[row_idx][idx] == background and next_region[row_idx][j] != background:
                        result[row_idx][idx] = next_region[row_idx][j]
                result[row_idx].extend(next_region[row_idx][overlap:])
            else:
                result[row_idx].extend(next_region[row_idx])

    return result
