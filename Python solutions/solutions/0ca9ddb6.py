def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 9x9 grid with sparse non-zero values (1, 2, 6, 8).
    2. Output preserves all original non-zero values in their positions.
    3. Value 1 gets surrounded by 7s in cross pattern (4-connected neighbors).
    4. Value 2 gets surrounded by 4s in corner pattern (diagonal neighbors).
    5. Values 6 and 8 remain unchanged with no surrounding pattern.
    6. New values (4s and 7s) are only placed in empty cells (value 0).

    Procedure:
    1. Create a copy of the input grid as the result.
    2. Iterate through each cell in the original grid.
    3. For cells with value 1, place 7s in the four orthogonally adjacent positions if they are empty.
    4. For cells with value 2, place 4s in the four diagonally adjacent positions if they are empty.
    5. Leave all other values unchanged.
    6. Return the modified grid.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy input

    # Find all non-zero values and their positions
    for r in range(rows):
        for c in range(cols):
            value = grid[r][c]

            if value == 1:
                # Surround with 7s in cross pattern (4-connected neighbors)
                cross_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for dr, dc in cross_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                        result[nr][nc] = 7

            elif value == 2:
                # Surround with 4s in corner pattern (diagonal neighbors)
                corner_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                for dr, dc in corner_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 0:
                        result[nr][nc] = 4

            # Values 8, 6, and others stay unchanged with no surrounding pattern

    return result
