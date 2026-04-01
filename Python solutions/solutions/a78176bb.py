def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with a diagonal line (non-0, non-5) and scattered 5s
    2. All diagonals go down-right with equation c = r + offset
    3. Output removes all 5s and adds new parallel diagonals
    4. The 5s indicate where/how to add new diagonals
    5. Spacing between diagonals is related to the width of the 5s region

    Procedure:
    1. Find the original diagonal color and offset (c - r constant)
    2. Find the 5s bounding box (min/max columns)
    3. Calculate spacing = width_of_5s + adjustment
    4. Add new diagonals based on 5s position relative to original diagonal
    5. Continue adding diagonals with regular spacing to fill the grid
    """

    n = len(grid)
    diagonal_color = None
    diagonal_offset = None
    min_col_5 = float('inf')
    max_col_5 = -1

    # Find diagonal color and offset, and 5s bounding box
    for r in range(n):
        for c in range(n):
            if grid[r][c] != 0 and grid[r][c] != 5:
                diagonal_color = grid[r][c]
                if diagonal_offset is None:
                    diagonal_offset = c - r
            if grid[r][c] == 5:
                min_col_5 = min(min_col_5, c)
                max_col_5 = max(max_col_5, c)

    if diagonal_color is None:
        return [[0] * n for _ in range(n)]

    width_5 = max_col_5 - min_col_5 + 1 if max_col_5 >= 0 else 0

    # Create output grid
    result = [[0] * n for _ in range(n)]

    # Determine spacing and new diagonal offsets
    # If original diagonal offset >= min_col_5, 5s are on/left of diagonal
    # Otherwise, 5s are right of diagonal

    if diagonal_offset >= min_col_5:
        # 5s are on/left of diagonal, create new diagonal on the left
        spacing = width_5 + 2
        new_offsets = [diagonal_offset, diagonal_offset - spacing]

        # Don't add a third diagonal for this case
    else:
        # 5s are right of diagonal, create new diagonal on the right
        spacing = width_5 + 1
        new_offsets = [diagonal_offset, min_col_5]

        # Check if we need a third diagonal for "wrap around"
        # Only add wrap around if the original diagonal doesn't start at row 0
        # Find the starting row of the original diagonal
        start_row_of_original = max(0, -diagonal_offset)

        if start_row_of_original > 0:
            # Original diagonal doesn't start at row 0, add wrap around
            # Find the last row where the rightmost diagonal is visible
            last_row_of_right_diag = n - 1 - min_col_5
            if last_row_of_right_diag < n - 1:
                # The wrap around diagonal should start at the row where the right diagonal ends
                wrap_row = last_row_of_right_diag
                # It should start at column 0, so offset = 0 - wrap_row
                wrap_offset = 0 - wrap_row
                new_offsets.append(wrap_offset)

    # Draw all diagonals
    for offset in new_offsets:
        for r in range(n):
            c = r + offset
            if 0 <= c < n:
                result[r][c] = diagonal_color

    return result
