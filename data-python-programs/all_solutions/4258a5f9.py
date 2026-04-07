def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing mostly 0s with some cells having value 5
    2. Output transforms each cell with value 5 into the center of a 3x3 square
    3. The 3x3 square is filled with 1s, except the center which remains 5
    4. All other cells that are not part of any 3x3 square remain 0

    Procedure:
    1. Create a copy of the input grid
    2. Find all positions with value 5
    3. For each 5, create a 3x3 square centered on that position
    4. Fill the 3x3 square with 1s, keeping the center as 5
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Create a copy

    # Find all positions with value 5
    fives = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives.append((r, c))

    # For each 5, create a 3x3 square
    for r, c in fives:
        # Fill 3x3 square centered at (r, c)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                # Check bounds
                if 0 <= nr < rows and 0 <= nc < cols:
                    if dr == 0 and dc == 0:
                        # Center remains 5
                        result[nr][nc] = 5
                    else:
                        # Surrounding cells become 1
                        result[nr][nc] = 1

    return result
