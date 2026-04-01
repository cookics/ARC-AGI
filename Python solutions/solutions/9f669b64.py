def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains exactly 3 non-7 colored regions
    2. Smallest region moves to edge or into a gap
    3. Wide/tall regions spanning middle (rows/cols 4-5) split
    4. Vertical stack → horizontal split at cols 4-5
    5. Horizontal arrangement → vertical split at rows 4-5

    Procedure:
    1. Extract all colored regions
    2. Determine arrangement (vertical/horizontal)
    3. Find smallest region and region to split
    4. Apply transformations
    """

    n, m = len(grid), len(grid[0])
    output = [[7] * m for _ in range(n)]

    # Extract regions
    regions = {}
    for r in range(n):
        for c in range(m):
            if grid[r][c] != 7:
                color = grid[r][c]
                if color not in regions:
                    regions[color] = []
                regions[color].append((r, c))

    # Find smallest region (break ties by preferring higher color value for stability)
    min_size = min(len(cells) for cells in regions.values())
    candidates = [c for c in regions.keys() if len(regions[c]) == min_size]
    smallest_color = min(candidates)  # Prefer lower color number

    # Determine arrangement
    all_rows, all_cols = [], []
    for cells in regions.values():
        for r, c in cells:
            all_rows.append(r)
            all_cols.append(c)

    in_mid_cols = sum(1 for c in all_cols if 3 <= c <= 6)
    in_mid_rows = sum(1 for r in all_rows if 2 <= r <= 7)

    vertical_stack = in_mid_cols > len(all_cols) * 0.6
    horizontal_layout = in_mid_rows > len(all_rows) * 0.6

    # Find region to split
    split_color = None
    split_dir = None

    if vertical_stack:
        # Look for widest region spanning cols 4-5
        max_w = 0
        for color, cells in regions.items():
            if color == smallest_color:
                continue
            cols = [c for _, c in cells]
            w = max(cols) - min(cols) + 1
            if w >= 4 and min(cols) <= 4 and max(cols) >= 5 and w > max_w:
                max_w = w
                split_color = color
                split_dir = 'horizontal'
    elif horizontal_layout:
        # Look for narrowest tall region spanning rows 4-5
        min_w = float('inf')
        for color, cells in regions.items():
            if color == smallest_color:
                continue
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            h = max(rows) - min(rows) + 1
            w = max(cols) - min(cols) + 1
            if h >= 4 and min(rows) <= 4 and max(rows) >= 5 and w < min_w:
                min_w = w
                split_color = color
                split_dir = 'vertical'

    # Transform smallest region
    smallest_cells = regions[smallest_color]
    srows = [r for r, _ in smallest_cells]
    scols = [c for _, c in smallest_cells]
    sh = max(srows) - min(srows) + 1
    sw = max(scols) - min(scols) + 1

    if split_color:
        if split_dir == 'horizontal':
            split_rows = [r for r, _ in regions[split_color]]
            if max(srows) < min(split_rows):
                row_off = (n - sh) - min(srows)
            else:
                row_off = -min(srows)
            col_off = 4 - min(scols)
        else:  # vertical
            split_cols = [c for _, c in regions[split_color]]
            if max(scols) < min(split_cols):
                col_off = (m - sw) - min(scols)
            else:
                col_off = -min(scols)
            row_off = 4 - min(srows)
    else:
        # No split - move to empty edge
        if vertical_stack:
            top_empty = all(grid[r][c] == 7 for r in range(min(srows))
                          for c in range(min(scols), max(scols)+1))
            row_off = -min(srows) if (top_empty and min(srows) > 0) else (n - sh) - min(srows)
            col_off = 0
        elif horizontal_layout:
            left_empty = all(grid[r][c] == 7
                           for r in range(min(srows), max(srows)+1)
                           for c in range(min(scols)))
            col_off = -min(scols) if (left_empty and min(scols) > 0) else (m - sw) - min(scols)
            row_off = 0
        else:
            # Default: move to bottom
            row_off = (n - sh) - min(srows)
            col_off = 0

    # Apply transformations
    for color, cells in regions.items():
        if color == smallest_color:
            for r, c in cells:
                nr, nc = r + row_off, c + col_off
                if 0 <= nr < n and 0 <= nc < m:
                    output[nr][nc] = color
        elif color == split_color:
            if split_dir == 'horizontal':
                for r, c in cells:
                    if c <= 4 and c - 1 >= 0:
                        output[r][c-1] = color
                    elif c >= 5 and c + 1 < m:
                        output[r][c+1] = color
            else:  # vertical
                for r, c in cells:
                    if r <= 4 and r - 1 >= 0:
                        output[r-1][c] = color
                    elif r >= 5 and r + 1 < n:
                        output[r+1][c] = color
        else:
            for r, c in cells:
                output[r][c] = color

    return output
