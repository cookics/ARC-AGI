def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing values 0, 5, and exactly one special value (e.g., 1, 2, 3)
    2. Output is the same grid with some 0s replaced by the special value
    3. The special value appears once in the input at some row position
    4. The transformation applies to rows with the same parity (even/odd) as the special value's row
    5. Within those rows, 0s at odd column indices are replaced with the special value

    Procedure:
    1. Find the special cell (non-0, non-5 value) and get its position and value
    2. Determine the parity of the special cell's row (even or odd)
    3. For all rows with the same parity, replace 0s at odd column positions with the special value
    4. Return the modified grid
    """

    # Create a copy of the grid to modify
    result = [row[:] for row in grid]

    # Find the special cell (non-0, non-5 value)
    special_value = None
    special_row = None

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != 0 and grid[r][c] != 5:
                special_value = grid[r][c]
                special_row = r
                break
        if special_value is not None:
            break

    # Determine the parity of the special cell's row
    target_parity = special_row % 2

    # For rows with the same parity, replace 0s with special value at odd columns
    for r in range(len(result)):
        if r % 2 == target_parity:  # Same parity as special cell's row
            for c in range(len(result[r])):
                if c % 2 == 1 and result[r][c] == 0:  # Odd column and value is 0
                    result[r][c] = special_value

    return result
