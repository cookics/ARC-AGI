def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid (list of lists) containing integers, mostly zeros with some non-zero values.
    2. Output is a transformed grid where certain rows are completely filled with a specific value.
    3. The pattern is: when a row has the same non-zero value at both the first and last positions, the entire row gets filled with that value.
    4. If the first and last values are different, or if either is zero, the row remains unchanged.
    5. Rows with all zeros remain unchanged.

    Procedure:
    1. Iterate through each row in the input grid.
    2. For each row, check if it has at least one element.
    3. Compare the first element (index 0) with the last element (index -1).
    4. Check if both elements are equal and non-zero.
    5. If both conditions are met, create a new row filled entirely with that value.
    6. If conditions are not met, keep the original row unchanged.
    7. Return the transformed grid.
    """
    result = []

    for row in grid:
        new_row = row.copy()

        # Check if first and last elements are the same and non-zero
        if len(row) > 0 and row[0] == row[-1] and row[0] != 0:
            # Fill entire row with that value
            fill_value = row[0]
            new_row = [fill_value] * len(row)

        result.append(new_row)

    return result
