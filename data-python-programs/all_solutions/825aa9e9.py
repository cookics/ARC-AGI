def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with separators dividing sections
    2. Within sections, values fall downward (gravity) per column
    3. Background values (most common) float up, patterns sink
    4. Last row in working range acts as floor

    Procedure:
    1. Find uniform columns (vertical separators) and uniform rows (floors)
    2. Group columns into sections between separators
    3. For each section, find background value
    4. Apply column-wise gravity: bg up, non-bg down, floor stays
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0]) if grid else 0
    result = [list(row) for row in grid]  # Ensure mutable copy

    # Uniform columns (all same value vertically)
    unif_cols = set()
    for c in range(cols):
        if len({grid[r][c] for r in range(rows)}) == 1:
            unif_cols.add(c)

    # Marker columns (rare values appearing multiple times in a row)
    mark_cols = set()
    for r in range(rows):
        cnt = Counter(grid[r])
        if len(cnt) > 1:
            rare = min(cnt, key=cnt.get)
            if 1 < cnt[rare] < cols:
                mark_cols.update(c for c in range(cols) if grid[r][c] == rare)

    sep_cols = unif_cols | mark_cols

    # Uniform rows (all same value horizontally)
    unif_rows = {r for r in range(rows) if len(set(grid[r])) == 1}
    work_rows = [r for r in range(rows) if r not in unif_rows]

    if not work_rows:
        return result

    # Build column sections
    secs = []
    st = None
    for c in range(cols):
        if c in sep_cols:
            if st is not None:
                secs.append((st, c - 1))
                st = None
        elif st is None:
            st = c
    if st is not None:
        secs.append((st, cols - 1))
    if not secs:
        secs = [(0, cols - 1)]

    # Process sections
    for c0, c1 in secs:
        # Section background
        all_vals = [grid[r][c] for r in work_rows for c in range(c0, c1 + 1)]
        if not all_vals:
            continue
        bg = Counter(all_vals).most_common(1)[0][0]

        # Gravity per column
        for c in range(c0, c1 + 1):
            col_vals = [grid[r][c] for r in work_rows]
            if len(col_vals) <= 1:
                continue

            # Last is floor
            flr = col_vals[-1]
            above = col_vals[:-1]

            # Split by background
            bg_list = [v for v in above if v == bg]
            nb_list = [v for v in above if v != bg]

            # New column: bg first, non-bg next, floor last
            new_vals = bg_list + nb_list + [flr]

            # Apply
            for idx, r in enumerate(work_rows):
                result[r][c] = new_vals[idx]

    return result
