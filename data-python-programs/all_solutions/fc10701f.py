def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    - Input has colored blocks (7s and 9s) and scattered 0s on a background of 6s
    - 7s move to where 9s are located
    - Original positions of both 7s and 9s become 6s
    - 2s appear at intersections based on 0 positions and colored block positions

    Procedure:
    1. Find all positions of 9s and 7s
    2. Create output grid by copying input
    3. Move 7s to 9 positions
    4. Erase original 9 and 7 positions (set to 6)
    5. Place 2s based on intersection of rows/columns with 0s and colored blocks
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # deep copy

    # Find positions of 9s and 7s
    nines = []
    sevens = []
    zeros = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 9:
                nines.append((r, c))
            elif grid[r][c] == 7:
                sevens.append((r, c))
            elif grid[r][c] == 0:
                zeros.append((r, c))

    # Move 7s to 9 positions and erase originals
    for r, c in nines:
        result[r][c] = 7

    for r, c in sevens:
        result[r][c] = 6

    for r, c in nines:
        if (r, c) not in sevens:  # Don't double-erase if 7 was moved here
            pass  # Already set to 7 above

    # Find rows and columns that have 0s
    rows_with_zeros = set(r for r, c in zeros)
    cols_with_zeros = set(c for r, c in zeros)

    # Find rows and columns that had colored blocks
    rows_with_colors = set(r for r, c in nines + sevens)
    cols_with_colors = set(c for r, c in nines + sevens)

    # Determine the pattern for placing 2s
    nine_rows = set(r for r, c in nines)
    nine_cols = set(c for r, c in nines)
    seven_rows = set(r for r, c in sevens)
    seven_cols = set(c for r, c in sevens)

    # Check if 9s and 7s share the same columns
    if nine_cols == seven_cols:
        # Case 1: Same columns - place 2s in (rows with 0s, cols with colors)
        for r in range(rows):
            for c in range(cols):
                if r in rows_with_zeros and c in cols_with_colors and result[r][c] == 6:
                    result[r][c] = 2
    else:
        # Case 2: Different columns - place 2s in (rows with colors, cols with 0s)
        # But only for columns that have 0s and are between the colored regions
        if nine_cols and seven_cols:
            min_color_col = min(min(nine_cols), min(seven_cols))
            max_color_col = max(max(nine_cols), max(seven_cols))

            for r in range(rows):
                for c in range(cols):
                    if (
                        r in rows_with_colors
                        and c in cols_with_zeros
                        and result[r][c] == 6
                        and min_color_col < c < max_color_col
                    ):
                        result[r][c] = 2

    return result
