def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 6-row grid, output is a 9-row grid with same width.
    2. Each column has a repeating periodic pattern that needs to be detected.
    3. Columns are extended by cycling through their shortest period.
    4. All 1s in the extended grid are replaced with 2s.
    5. 0s remain unchanged in the transformation.

    Procedure:
    1. Extract each column from the input grid.
    2. Find the shortest repeating period for each column.
    3. Extend each column to 9 rows by cycling through its period.
    4. Convert the extended columns back to grid format.
    5. Replace all 1s with 2s in the final result.
    """

    def find_period(seq):
        """Find the shortest period of a sequence"""
        n = len(seq)
        for period in range(1, n + 1):
            is_periodic = True
            for i in range(n):
                if seq[i] != seq[i % period]:
                    is_periodic = False
                    break
            if is_periodic:
                return period
        return n

    def extend_column(col, target_length):
        """Extend a column to target length using its periodic pattern"""
        period = find_period(col)
        extended = []
        for i in range(target_length):
            extended.append(col[i % period])
        return extended

    # Extract columns
    num_cols = len(grid[0])
    columns = []
    for j in range(num_cols):
        col = [grid[i][j] for i in range(len(grid))]
        columns.append(col)

    # Extend each column to 9 rows
    extended_columns = []
    for col in columns:
        extended_col = extend_column(col, 9)
        extended_columns.append(extended_col)

    # Convert back to grid format and change 1s to 2s
    result = []
    for i in range(9):
        row = []
        for j in range(num_cols):
            val = extended_columns[j][i]
            if val == 1:
                val = 2
            row.append(val)
        result.append(row)

    return result
