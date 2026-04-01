def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with 0s and non-zero values
    2. Output replaces some 0s with 3s forming a cross pattern
    3. Vertical stripe: contiguous columns that are entirely 0
    4. Horizontal stripes: rows that are entirely 0 OR entirely 0 from vertical stripe start onwards
    5. Only cells that are 0 become 3

    Procedure:
    1. Find contiguous block of entirely-0 columns (vertical stripe)
    2. Find entirely-0 rows (horizontal stripes)
    3. Find rows that are entirely 0 from vertical stripe start to end (extended horizontal stripes)
    4. Fill all these with 3s
    """

    if not grid or not grid[0]:
        return grid

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find the largest rectangular block of consecutive zeros
    # This will define our vertical stripe (columns) and horizontal stripe (rows)

    def find_largest_zero_rect():
        """Find the largest rectangular region of all zeros"""
        max_area = 0
        best_rect = None

        # Try all possible row ranges
        for r1 in range(rows):
            for r2 in range(r1, rows):
                # For this row range, find longest consecutive zero columns
                col_start = -1
                for c in range(cols):
                    all_zero = all(grid[r][c] == 0 for r in range(r1, r2 + 1))
                    if all_zero:
                        if col_start == -1:
                            col_start = c
                    else:
                        if col_start != -1:
                            area = (r2 - r1 + 1) * (c - col_start)
                            if area > max_area:
                                max_area = area
                                best_rect = (r1, r2, col_start, c - 1)
                            col_start = -1
                if col_start != -1:
                    area = (r2 - r1 + 1) * (cols - col_start)
                    if area > max_area:
                        max_area = area
                        best_rect = (r1, r2, col_start, cols - 1)

        return best_rect

    rect = find_largest_zero_rect()
    if rect is None:
        return grid

    h_start, h_end, c_start, c_end = rect

    # Vertical stripe is the column range, excluding first and last
    if c_end - c_start >= 2:
        v_start = c_start + 1
        v_end = c_end - 1
    else:
        v_start = c_start
        v_end = c_end

    # Find entirely-0 rows (full horizontal stripes)
    zero_rows_full = set()
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            zero_rows_full.add(r)

    # Find consecutive blocks of rows with zeros from v_start onwards
    # Then select only interior rows of each block for right extension
    zero_rows_right = set()
    if v_start < cols:
        # Find all rows with zeros from v_start to end (excluding entirely-0 rows)
        candidate_rows = []
        for r in range(rows):
            if r not in zero_rows_full and all(grid[r][c] == 0 for c in range(v_start, cols)):
                candidate_rows.append(r)

        # Group into consecutive blocks
        if candidate_rows:
            blocks = []
            block_start = candidate_rows[0]
            block_end = candidate_rows[0]

            for i in range(1, len(candidate_rows)):
                if candidate_rows[i] == block_end + 1:
                    block_end = candidate_rows[i]
                else:
                    blocks.append((block_start, block_end))
                    block_start = candidate_rows[i]
                    block_end = candidate_rows[i]
            blocks.append((block_start, block_end))

            # For each block, select interior rows (exclude first and last)
            for block_start, block_end in blocks:
                if block_end - block_start >= 2:  # At least 3 rows
                    for r in range(block_start + 1, block_end):
                        zero_rows_right.add(r)
                elif block_end - block_start == 1:  # Exactly 2 rows, no interior
                    pass
                # Single row blocks are not extended

    # Find rows with left extension: 0s from left edge to v_end
    # Apply same logic: find consecutive blocks and select interior rows
    # BUT exclude rows that are at the boundary of the zero rectangle
    zero_rows_left = set()
    if v_end >= 0:
        # Find all rows with zeros from 0 to v_end (excluding entirely-0, right-extended, and boundary rows)
        candidate_rows = []
        for r in range(rows):
            if r == h_start or r == h_end:
                continue  # Skip boundary rows
            if r not in zero_rows_full and r not in zero_rows_right and all(grid[r][c] == 0 for c in range(0, v_end + 1)):
                candidate_rows.append(r)

        # Group into consecutive blocks
        if candidate_rows:
            blocks = []
            block_start = candidate_rows[0]
            block_end = candidate_rows[0]

            for i in range(1, len(candidate_rows)):
                if candidate_rows[i] == block_end + 1:
                    block_end = candidate_rows[i]
                else:
                    blocks.append((block_start, block_end))
                    block_start = candidate_rows[i]
                    block_end = candidate_rows[i]
            blocks.append((block_start, block_end))

            # For each block, select interior rows
            for block_start, block_end in blocks:
                if block_end - block_start >= 2:
                    for r in range(block_start + 1, block_end):
                        zero_rows_left.add(r)

    # Fill vertical stripe (only within the rectangle row range or extended rows, excluding boundary rows)
    if v_start != -1 and v_start < cols:
        for c in range(v_start, v_end + 1):
            for r in range(rows):
                # Only fill if: (1) within rectangle row range (not boundaries), OR (2) in an extended row
                in_core = h_start < r < h_end  # Exclude boundary rows
                if (in_core or r in zero_rows_full or r in zero_rows_right or r in zero_rows_left):
                    if grid[r][c] == 0:
                        result[r][c] = 3

    # Fill entirely-0 rows (already all 0, so make them all 3)
    for r in zero_rows_full:
        for c in range(cols):
            result[r][c] = 3

    # Fill rows with right extension (from v_start to end)
    for r in zero_rows_right:
        for c in range(v_start, cols):
            if grid[r][c] == 0:
                result[r][c] = 3

    # Fill rows with left extension (from 0 to v_end)
    for r in zero_rows_left:
        for c in range(0, v_end + 1):
            if grid[r][c] == 0:
                result[r][c] = 3

    return result
