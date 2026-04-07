def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with numbers where one row consists entirely of 9s (acting as a divider).
    2. The 6s in the input indicate where 2s should be placed in the output.
    3. The 5s should be placed at the reflected positions of the 6s across the divider row.
    4. All existing 2s and 5s in the input are ignored.

    Procedure:
    1. Find the row that consists entirely of 9s (divider row)
    2. Create output grid filled with 7s, keeping the divider row as 9s
    3. For each 6 in the input:
       - Place a 2 at that position
       - Place a 5 at the reflected position across the divider
    4. Handle conflicts when multiple 6s exist in the same column
    """

    # Find dimensions
    rows = len(grid)
    cols = len(grid[0])

    # Find the divider row (row of all 9s)
    divider_row = -1
    for i in range(rows):
        if all(cell == 9 for cell in grid[i]):
            divider_row = i
            break

    # Initialize output grid with all 7s
    result = [[7 for _ in range(cols)] for _ in range(rows)]

    # Keep the divider row as all 9s
    for j in range(cols):
        result[divider_row][j] = 9

    # Group 6s by column to handle conflicts
    column_sixes = {}
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 6:
                if j not in column_sixes:
                    column_sixes[j] = []
                column_sixes[j].append(i)

    # Process each column
    for col, six_rows in column_sixes.items():
        if len(six_rows) == 1:
            # Single 6: place 2 and reflected 5
            six_row = six_rows[0]
            result[six_row][col] = 2
            reflected_row = 2 * divider_row - six_row
            if 0 <= reflected_row < rows:
                result[reflected_row][col] = 5
        else:
            # Multiple 6s in same column: choose based on existing 2s/5s in this column
            existing_twos = [i for i in range(rows) if grid[i][col] == 2]
            existing_fives = [i for i in range(rows) if grid[i][col] == 5]

            if existing_twos and existing_fives:
                # Both 2s and 5s exist: choose based on position of existing 2 relative to 6s
                min_six_row = min(six_rows)
                max_six_row = max(six_rows)
                existing_two_below_divider = [
                    row for row in existing_twos if row > divider_row
                ]

                if existing_two_below_divider:
                    # Check if existing 2 is above or below the 6s
                    if any(
                        two_row < min_six_row for two_row in existing_two_below_divider
                    ):
                        # Existing 2 is above the 6s: use the furthest 6 from divider
                        target_six_row = max(
                            six_rows, key=lambda r: abs(r - divider_row)
                        )
                    else:
                        # Existing 2 is below the 6s: use the closest 6 to divider
                        target_six_row = min(
                            six_rows, key=lambda r: abs(r - divider_row)
                        )
                else:
                    # No 2s below divider, use furthest from divider
                    target_six_row = max(six_rows, key=lambda r: abs(r - divider_row))
            else:
                # Default: use closest to divider
                target_six_row = min(six_rows, key=lambda r: abs(r - divider_row))

            result[target_six_row][col] = 2
            reflected_row = 2 * divider_row - target_six_row
            if 0 <= reflected_row < rows:
                result[reflected_row][col] = 5

    return result
