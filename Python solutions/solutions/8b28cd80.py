def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3×3 grid with exactly one non-zero cell at position (r_in, c_in)
    2. Output is a 9×9 grid with a hierarchical fractal pattern
    3. For each output cell at (r, c), we compute:
       - Block coordinates: (r//3, c//3) - which 3×3 block
       - Within-block coordinates: (r%3, c%3) - position within block
    4. The fill rule involves 4 disjunctive conditions that create the pattern

    Procedure:
    1. Find the non-zero cell position (r_in, c_in) and value
    2. Apply the complex formula to determine which cells to fill
    """

    # Find the non-zero cell
    r_in, c_in, value = None, None, None
    for r in range(3):
        for c in range(3):
            if grid[r][c] != 0:
                r_in, c_in, value = r, c, grid[r][c]
                break
        if value is not None:
            break

    # Create 9×9 output
    result = [[0] * 9 for _ in range(9)]

    # Apply the fractal fill rule
    for r in range(9):
        for c in range(9):
            block_r = r // 3
            block_c = c // 3
            within_r = r % 3
            within_c = c % 3

            # Four conditions that determine when cell is NOT filled
            cond1 = (within_r == r_in) and (block_c == c_in)
            cond2 = (within_c == c_in) and (block_r == r_in)
            cond3 = (block_r != r_in and within_c == r_in and block_c != c_in and
                     not (within_r == r_in and block_c == r_in))
            cond4 = (block_c != c_in and within_r == c_in and block_r != r_in and
                     not (within_c == c_in and block_r == c_in))

            # Fill if none of the conditions are true
            if not (cond1 or cond2 or cond3 or cond4):
                result[r][c] = value

    return result
