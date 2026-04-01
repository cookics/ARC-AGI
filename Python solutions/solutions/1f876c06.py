def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid containing mostly zeros with some scattered non-zero numbers.
    2. Each non-zero value appears exactly twice in the input grid.
    3. Output connects each pair of identical non-zero numbers with a perfect diagonal line.
    4. The diagonal line is filled with the same non-zero value throughout its path.
    5. Diagonal lines move one step diagonally at a time, either up-left to down-right or up-right to down-left.
    6. Original non-zero positions are preserved in their exact locations.

    Procedure:
    1. Create a copy of the input grid to avoid modifying the original.
    2. Scan the entire grid to find all non-zero positions and group them by their values.
    3. For each value that appears exactly twice, identify the two positions.
    4. Calculate the diagonal direction between the two positions (row and column increments).
    5. Draw a diagonal line by iteratively moving from the first position toward the second position.
    6. Fill each position along the diagonal path with the corresponding value.
    7. Return the modified grid with all diagonal connections drawn.
    """

    result = [row[:] for row in grid]  # Copy the grid

    # Find all non-zero positions grouped by value
    positions_by_value = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] != 0:
                value = grid[r][c]
                if value not in positions_by_value:
                    positions_by_value[value] = []
                positions_by_value[value].append((r, c))

    # For each value with exactly 2 positions, draw diagonal line
    for value, positions in positions_by_value.items():
        if len(positions) == 2:
            r1, c1 = positions[0]
            r2, c2 = positions[1]

            # Determine direction
            dr = 1 if r2 > r1 else -1
            dc = 1 if c2 > c1 else -1

            # Draw diagonal line
            r, c = r1, c1
            while True:
                result[r][c] = value
                if r == r2 and c == c2:
                    break
                r += dr
                c += dc

    return result
