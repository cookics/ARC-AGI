def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has L-shaped 1-regions with small anomalies
    2. Need to fix: (a) vertical gaps, (b) horizontal gap columns, (c) edge removals
    3. Vertical gap: 8 with 1s above and below in same column
    4. For rows with horizontal gaps: remove edge 1s from certain columns
    5. Extend gap columns downward into sparse rows

    Procedure:
    1. Find and fill vertical gaps (8s surrounded by 1s in column)
    2. Find horizontal gaps in dense rows
    3. For rows with gaps: remove top-edge 1s from certain columns
    4. Extend gap columns downward
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Step 1: Fill vertical gaps - 8s that have 1s both above and below
    for r in range(1, rows - 1):
        for c in range(cols):
            if result[r][c] == 8:
                # Check if surrounded vertically by 1s
                if result[r-1][c] == 1 and result[r+1][c] == 1:
                    result[r][c] = 1

    # Step 2: Find horizontal gaps in rows
    # A gap is an 8 surrounded by 1s in the same row
    gaps_by_row = {}
    for r in range(rows):
        row_ones = sum(1 for val in grid[r] if val == 1)
        if row_ones >= 8:  # Dense row
            for c in range(1, cols - 1):
                if grid[r][c] == 8:
                    # Check if it's a gap (1s on both sides)
                    left_has_1 = any(grid[r][cc] == 1 for cc in range(max(0, c-3), c))
                    right_has_1 = any(grid[r][cc] == 1 for cc in range(c+1, min(cols, c+4)))
                    if left_has_1 and right_has_1:
                        if r not in gaps_by_row:
                            gaps_by_row[r] = []
                        gaps_by_row[r].append(c)

    # Step 3: For rows with gaps, find and remove edge cells from columns
    for gap_row, gap_cols in gaps_by_row.items():
        # Find columns that have 1 at gap_row and are at top edge
        for c in range(cols):
            if grid[gap_row][c] == 1:
                # Check if this is at top edge of a column run
                is_top_edge = (gap_row == 0 or grid[gap_row-1][c] == 8)
                if is_top_edge:
                    # Check if column has 1s below
                    has_below = any(grid[rr][c] == 1 for rr in range(gap_row+1, min(rows, gap_row+6)))
                    if has_below:
                        result[gap_row][c] = 8

    # Step 4: Extend gap columns downward into sparse rows
    for gap_row, gap_cols in gaps_by_row.items():
        for gap_col in gap_cols:
            # Extend downward until we find a row with 1s or hit bottom
            for r in range(gap_row + 1, rows):
                row_ones = sum(1 for val in result[r] if val == 1)
                if row_ones <= 5 and result[r][gap_col] == 8:  # Sparse row
                    # Check if there are 1s nearby in this column
                    nearby = any(result[rr][gap_col] == 1 for rr in range(max(0, r-2), min(rows, r+3)))
                    if nearby:
                        result[r][gap_col] = 1

    return result
