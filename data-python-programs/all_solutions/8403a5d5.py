def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with all zeros except one non-zero value in the bottom row
    2. Output is a 10x10 grid with vertical lines of that value
    3. Vertical lines start at the column where the value appears and continue every 2 columns to the right
    4. Value 5 appears in gaps between consecutive vertical lines
    5. The 5s alternate placement: first gap in row 0, second gap in row 9, third gap in row 0, etc.
    6. If the last vertical line doesn't reach column 9, a 5 is placed at (0, 9)

    Procedure:
    1. Find the non-zero value and its column position in the last row
    2. Create vertical lines at start_col, start_col+2, start_col+4, etc., filled with that value
    3. Identify gaps (columns between consecutive vertical lines)
    4. Place 5s in gaps alternating between row 0 (even-indexed gaps) and row 9 (odd-indexed gaps)
    5. If last vertical line is before column 9, place a 5 at (0, 9)
    """

    # Find the starting position and value
    start_col = -1
    value = 0
    for col in range(10):
        if grid[9][col] != 0:
            start_col = col
            value = grid[9][col]
            break

    # Create result grid
    result = [[0 for _ in range(10)] for _ in range(10)]

    # Fill vertical lines starting from start_col, every 2 columns
    filled_cols = []
    col = start_col
    while col <= 9:
        filled_cols.append(col)
        for row in range(10):
            result[row][col] = value
        col += 2

    # Find gaps between consecutive filled columns
    gaps = []
    for i in range(len(filled_cols) - 1):
        gap_col = filled_cols[i] + 1
        if gap_col < filled_cols[i + 1]:
            gaps.append(gap_col)

    # Place 5s in gaps, alternating between row 0 and row 9
    for i, gap_col in enumerate(gaps):
        if i % 2 == 0:  # Even index -> row 0
            result[0][gap_col] = 5
        else:  # Odd index -> row 9
            result[9][gap_col] = 5

    # If last filled column < 9, add 5 at column 9 in row 0
    if filled_cols[-1] < 9:
        result[0][9] = 5

    return result
