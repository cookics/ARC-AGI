def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid divided into column segments by separator columns
    2. Within each segment, rows with same background share all 8 column positions
    3. Separator rows divide but don't affect row grouping

    Procedure:
    1. Find column separators and divide into segments
    2. For each segment, infer background for each row
    3. Group rows by background within segment
    4. Collect and apply all 8 column positions per group
    """
    from collections import defaultdict, Counter

    H, W = len(grid), len(grid[0])

    # Find separator columns
    sep_cols = []
    for c in range(W):
        if all(grid[r][c] == grid[0][c] != 8 for r in range(H)):
            sep_cols.append(c)

    # Create column ranges
    col_ranges = []
    prev = -1
    for c in sep_cols + [W]:
        if c > prev + 1:
            col_ranges.append((prev + 1, c))
        prev = c

    # Build output
    output = [[grid[r][c] for c in range(W)] for r in range(H)]

    # Process each column segment
    for c1, c2 in col_ranges:
        # Infer background for each row in this segment
        backgrounds = {}
        for r in range(H):
            bg = [grid[r][c] for c in range(c1, c2)]
            for c_idx in range(len(bg)):
                if bg[c_idx] == 8:
                    # Collect non-8 values from other rows at this column
                    cands = [grid[r2][c1 + c_idx] for r2 in range(H) if grid[r2][c1 + c_idx] != 8]
                    if cands:
                        bg[c_idx] = Counter(cands).most_common(1)[0][0]
            backgrounds[r] = tuple(bg)

        # Group rows by background
        row_groups = defaultdict(list)
        for r in range(H):
            row_groups[backgrounds[r]].append(r)

        # Collect 8 positions per group
        for pattern, rows in row_groups.items():
            eights = set()
            for r in rows:
                for c_idx, c in enumerate(range(c1, c2)):
                    if grid[r][c] == 8:
                        eights.add(c_idx)

            # Apply to all rows in group
            for r in rows:
                for c_idx in range(len(pattern)):
                    c = c1 + c_idx
                    if c_idx in eights:
                        output[r][c] = 8
                    else:
                        output[r][c] = pattern[c_idx]

    return output
