def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid containing 0s and 5s.
    2. Output is a modified grid where specific rows get transformed based on a template pattern.
    3. Row 0 serves as a template pattern containing positions where 5s appear.
    4. For any row that has a 5 in the rightmost column (column 9), the template pattern is applied.
    5. When applying the template, all 5s in the template become 2s in the target row.
    6. The original 5 at position 9 in the target row is preserved.
    7. All other rows remain unchanged.

    Procedure:
    1. Extract the template pattern from row 0.
    2. Iterate through each row from 1 onwards.
    3. Check if the current row has a 5 in column 9.
    4. If yes, apply the template transformation where 5s become 2s, keeping the original 5 at column 9.
    5. If no, keep the row unchanged.
    6. Return the modified grid.
    """

    result = [row[:] for row in grid]  # Create deep copy

    # Get template from row 0
    template = grid[0][:]

    # For each row after the first, check if it has a 5 in the last column
    for i in range(1, len(grid)):
        if grid[i][9] == 5:  # Row has 5 in rightmost column
            # Apply template but replace 5s with 2s
            for j in range(9):  # Don't modify the last column
                if template[j] == 5:
                    result[i][j] = 2
                else:
                    result[i][j] = template[j]
            # Keep the original 5 in the last column
            result[i][9] = 5

    return result
