def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Looking at the training examples, certain 4s get converted to 8s in the output.
    2. The pattern is that cross-shaped formations of 4s get converted to 8s.
    3. A cross pattern consists of a center 4 with 4s in all four orthogonal directions (up, down, left, right).

    Procedure:
    1. Make a copy of the input grid
    2. For each cell, check if it's the center of a cross pattern of 4s
    3. If it is, mark all 5 cells of the cross pattern for conversion
    4. Convert all marked cells from 4 to 8
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find all cross patterns
    to_convert = set()

    for r in range(1, rows - 1):  # Skip edges since cross needs neighbors
        for c in range(1, cols - 1):
            if (
                grid[r][c] == 4
                and grid[r - 1][c] == 4  # above
                and grid[r + 1][c] == 4  # below
                and grid[r][c - 1] == 4  # left
                and grid[r][c + 1] == 4
            ):  # right
                # Mark all 5 cells of the cross for conversion
                to_convert.add((r, c))  # center
                to_convert.add((r - 1, c))  # above
                to_convert.add((r + 1, c))  # below
                to_convert.add((r, c - 1))  # left
                to_convert.add((r, c + 1))  # right

    # Convert marked cells
    for r, c in to_convert:
        result[r][c] = 8

    return result
