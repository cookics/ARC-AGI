def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing a 3x3 block of non-zero values surrounded by zeros.
    2. Output is the same grid with values projected from each position in the 3x3 block.
    3. Each position in the 3x3 pattern projects its value in a specific direction.
    4. Top row cells project upward and diagonally upward from their positions.
    5. Middle row cells project horizontally (left/right) from their positions.
    6. Bottom row cells project downward and diagonally downward from their positions.
    7. The center cell remains in its original position without projection.

    Procedure:
    1. Locate the 3x3 pattern of non-zero values in the input grid.
    2. Extract the values from each position in the 3x3 pattern.
    3. For each position, determine the projection direction based on its location in the pattern.
    4. Project each value along its designated direction until reaching grid boundaries.
    5. Return the modified grid with all projected values.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[grid[i][j] for j in range(cols)] for i in range(rows)]

    # Find the 3x3 pattern
    pattern_r, pattern_c = None, None
    for r in range(rows - 2):
        for c in range(cols - 2):
            if all(grid[r + i][c + j] != 0 for i in range(3) for j in range(3)):
                pattern_r, pattern_c = r, c
                break
        if pattern_r is not None:
            break

    if pattern_r is None:
        return result

    # Extract the 3x3 pattern
    pattern = [[grid[pattern_r + i][pattern_c + j] for j in range(3)] for i in range(3)]

    # Project each cell in the pattern according to its position
    for i in range(3):
        for j in range(3):
            val = pattern[i][j]
            start_r = pattern_r + i
            start_c = pattern_c + j

            # Determine projection directions based on position
            if i == 0:  # Top row
                if j == 0:  # Top-left: project up-left diagonal
                    for k in range(1, max(rows, cols)):
                        r, c = start_r - k, start_c - k
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = val
                elif j == 1:  # Top-middle: project upward
                    for r in range(start_r):
                        result[r][start_c] = val
                else:  # Top-right: project up-right diagonal
                    for k in range(1, max(rows, cols)):
                        r, c = start_r - k, start_c + k
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = val

            elif i == 1:  # Middle row
                if j == 0:  # Middle-left: project leftward
                    for c in range(start_c):
                        result[start_r][c] = val
                elif j == 1:  # Center: keep in place
                    pass
                else:  # Middle-right: project rightward
                    for c in range(start_c + 1, cols):
                        result[start_r][c] = val

            else:  # Bottom row
                if j == 0:  # Bottom-left: project down-left diagonal
                    for k in range(1, max(rows, cols)):
                        r, c = start_r + k, start_c - k
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = val
                elif j == 1:  # Bottom-middle: project downward
                    for r in range(start_r + 1, rows):
                        result[r][start_c] = val
                else:  # Bottom-right: project down-right diagonal
                    for k in range(1, max(rows, cols)):
                        r, c = start_r + k, start_c + k
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = val

    return result
