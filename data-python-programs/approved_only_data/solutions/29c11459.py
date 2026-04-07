def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with 5 rows and 11 columns containing integers.
    2. Output is the same grid structure but with certain rows modified.
    3. Pattern: rows with non-zero values at both first column (index 0) and last column (index 10) get filled.
    4. Filling pattern: positions 0-4 get the left value, position 5 gets value 5, positions 6-10 get the right value.
    5. Other rows remain unchanged from the input.

    Procedure:
    1. Create a copy of the input grid to avoid modifying the original.
    2. Iterate through each row in the grid.
    3. Check if the row has non-zero values at both position 0 and position 10.
    4. If condition is met, fill positions 0-4 with the left value (row[0]).
    5. Set position 5 to the value 5.
    6. Fill positions 6-10 with the right value (row[10]).
    7. Return the modified grid.
    """

    # Create a copy of the input grid
    result = [row[:] for row in grid]

    # Check each row
    for i in range(len(result)):
        row = result[i]

        # Check if row has non-zero values at positions 0 and 10
        assert len(row) == 11, f"Expected row length 11, got {len(row)}"

        if row[0] != 0 and row[10] != 0:
            left_value = row[0]
            right_value = row[10]

            # Fill the row according to the pattern
            # Positions 0-4: left value
            for j in range(5):
                row[j] = left_value

            # Position 5: value 5
            row[5] = 5

            # Positions 6-10: right value
            for j in range(6, 11):
                row[j] = right_value

    return result
