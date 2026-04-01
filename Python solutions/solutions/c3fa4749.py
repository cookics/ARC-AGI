def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Find rectangular blocks with dominant value (>75%)
    2. Within blocks, process rows/columns for outlier sequences
    3. If row/column starts with outlier(s), fill with first value
    4. Iterate multiple times to handle cascading changes

    Procedure:
    1. Find dominant rectangular blocks
    2. For rows/columns starting with outliers, fill with first value
    3. Repeat until no changes
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Iterate to handle cascading changes
    for _ in range(5):
        changed = False

        # Find dominant blocks
        blocks = []
        for h in range(3, min(12, rows + 1)):
            for w in range(3, min(12, cols + 1)):
                for r1 in range(rows - h + 1):
                    for c1 in range(cols - w + 1):
                        r2, c2 = r1 + h - 1, c1 + w - 1

                        region_vals = [result[r][c] for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]
                        counts = Counter(region_vals)
                        if len(counts) < 2:
                            continue

                        dominant, dom_count = counts.most_common(1)[0]
                        if dom_count / len(region_vals) >= 0.78:
                            blocks.append((r1, c1, r2, c2, dominant))

        # Process each block
        for r1, c1, r2, c2, dominant in blocks:
            # Process rows - find ALL contiguous outlier sequences
            for r in range(r1, r2 + 1):
                row_vals = [result[r][c] for c in range(c1, c2 + 1)]

                i = 0
                while i < len(row_vals):
                    if row_vals[i] != dominant:
                        # Found outlier sequence start
                        j = i
                        while j < len(row_vals) and row_vals[j] != dominant:
                            j += 1

                        # Fill if sequence has at least 2 outliers
                        if j - i >= 2:
                            fill_val = row_vals[i]
                            for k in range(i, j):
                                if result[r][c1 + k] != fill_val:
                                    result[r][c1 + k] = fill_val
                                    changed = True

                        i = j
                    else:
                        i += 1

            # Process columns - find ALL contiguous outlier sequences
            for c in range(c1, c2 + 1):
                col_vals = [result[r][c] for r in range(r1, r2 + 1)]

                i = 0
                while i < len(col_vals):
                    if col_vals[i] != dominant:
                        # Found outlier sequence start
                        j = i
                        while j < len(col_vals) and col_vals[j] != dominant:
                            j += 1

                        # Fill if sequence has at least 2 outliers
                        if j - i >= 2:
                            fill_val = col_vals[i]
                            for k in range(i, j):
                                if result[r1 + k][c] != fill_val:
                                    result[r1 + k][c] = fill_val
                                    changed = True

                        i = j
                    else:
                        i += 1

        if not changed:
            break

    return result
