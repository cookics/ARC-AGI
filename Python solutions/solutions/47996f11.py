def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Let me try vertical symmetry as a final attempt.
    Maybe the grid should be symmetric across a horizontal center line.

    Procedure:
    1. For each 6, find its vertically mirrored counterpart
    2. If the counterpart is not 6, use that value
    3. Keep iterating until no more changes are made
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Keep iterating until no more changes are made
    changed = True
    while changed:
        changed = False
        for i in range(rows):
            for j in range(cols):
                if result[i][j] == 6:
                    # Find vertically mirrored position (across horizontal center)
                    mirror_i = rows - 1 - i

                    if result[mirror_i][j] != 6:
                        result[i][j] = result[mirror_i][j]
                        changed = True

                    # Also try from original grid
                    elif grid[mirror_i][j] != 6:
                        result[i][j] = grid[mirror_i][j]
                        changed = True

    return result
