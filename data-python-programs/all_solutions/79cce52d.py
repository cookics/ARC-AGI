def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 7x7 grid with value 7 at position (0,0)
    2. Value 2 appears at positions (0,c) and (r,0) marking boundaries
    3. Output is a 6x6 grid formed by removing row 0 and column 0, then applying transformations
    4. The transformation depends on the positions (r,c) of the 2 markers
    5. The grid is divided into quadrants at position (r-1, c-1) after removing first row/col
    6. Elements from bottom-right sections are moved toward top-left

    Procedure:
    1. Find positions r and c where grid[r][0]==2 and grid[0][c]==2
    2. Extract the 6x6 inner grid (rows 1-6, cols 1-6)
    3. Apply transformation based on (r,c):
       - For r=1, c=1: No transformation (identity)
       - For r=c: Move last (r-1) rows/cols to front (symmetric permutation)
       - For r≠c: Apply row permutation and column transformation based on quadrants
    """

    # Find positions of the 2 markers
    r, c = 0, 0
    for i in range(len(grid)):
        if grid[i][0] == 2:
            r = i
            break
    for j in range(len(grid[0])):
        if grid[0][j] == 2:
            c = j
            break

    # Extract 6x6 grid by removing row 0 and column 0
    extracted = []
    for i in range(1, 7):
        row = []
        for j in range(1, 7):
            row.append(grid[i][j])
        extracted.append(row)

    # Apply transformation based on r and c
    if r == 1 and c == 1:
        # No transformation needed - just return the extracted grid
        return extracted

    # Calculate split points in the extracted grid
    split_r = r - 1  # Row split point (0-indexed)
    split_c = c - 1  # Column split point (0-indexed)

    # For symmetric cases where r == c, use simple permutation
    if r == c:
        # Move last (r-1) elements to the front for both rows and columns
        n_move = r - 1
        row_perm = list(range(6 - n_move, 6)) + list(range(6 - n_move))

        result = [[0] * 6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                result[i][j] = extracted[row_perm[i]][row_perm[j]]
        return result

    # For asymmetric cases, apply row permutation and column transformation
    # Row permutation: move last (split_r) rows to front
    row_perm = list(range(6 - split_r, 6)) + list(range(6 - split_r))

    # Apply row permutation first
    reordered = [extracted[row_perm[i]] for i in range(6)]

    # Apply column transformation: split at split_c and rearrange
    result = [[0] * 6 for _ in range(6)]

    for i in range(6):
        # Determine which input row this came from
        source_row = row_perm[i]

        # Split the row at split_c
        left = reordered[i][:split_c]
        right = reordered[i][split_c:]

        # Determine if we should reverse the right part
        # Based on analysis: rows from certain positions get reversed
        # This appears to depend on whether the source row is within certain bounds
        if source_row >= split_r and source_row < 6 - split_r + split_r:
            # Don't reverse for rows in middle section
            result[i] = right + left
        else:
            # Reverse right part for other rows
            result[i] = right[::-1] + left

    return result
