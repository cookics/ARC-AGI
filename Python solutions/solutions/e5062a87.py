def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with values 0, 2, and 5
    2. Transformation reflects existing 2s across vertical axis at col 5
    3. After reflection, cast rays from all 2s to fill adjacent 0s
    4. Rays extend in 4 directions, stopping at 5s or other 2s

    Procedure:
    1. Find all existing 2 positions
    2. Reflect each 2 across vertical center (col 4/5 boundary)
    3. If reflected position is 0, mark it as 2
    4. Cast rays from all 2s (original + reflected) in 4 directions
    5. Rays fill consecutive 0s until hitting a 5 or 2
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all positions with 2s
    twos = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.append((r, c))

    # Reflect 2s across vertical center (between cols 4 and 5)
    # For col c: reflected_col = 9 - c
    reflected_positions = []
    for r, c in twos:
        reflected_c = cols - 1 - c
        if reflected_c != c and 0 <= reflected_c < cols:
            if result[r][reflected_c] == 0:
                result[r][reflected_c] = 2
                reflected_positions.append((r, reflected_c))

    # Combine original and reflected positions
    all_twos = twos + reflected_positions

    # Cast rays from all 2s to fill adjacent 0s
    # Use a queue to process positions that might cast rays
    to_process = list(all_twos)
    processed = set()

    while to_process:
        r, c = to_process.pop(0)
        if (r, c) in processed:
            continue
        processed.add((r, c))

        # Cast rays in 4 directions
        # North
        for rr in range(r - 1, -1, -1):
            if result[rr][c] == 0:
                result[rr][c] = 2
                to_process.append((rr, c))
            elif result[rr][c] == 2:
                break  # Stop at another 2
            elif result[rr][c] == 5:
                break  # Stop at 5

        # South
        for rr in range(r + 1, rows):
            if result[rr][c] == 0:
                result[rr][c] = 2
                to_process.append((rr, c))
            elif result[rr][c] == 2:
                break
            elif result[rr][c] == 5:
                break

        # West
        for cc in range(c - 1, -1, -1):
            if result[r][cc] == 0:
                result[r][cc] = 2
                to_process.append((r, cc))
            elif result[r][cc] == 2:
                break
            elif result[r][cc] == 5:
                break

        # East
        for cc in range(c + 1, cols):
            if result[r][cc] == 0:
                result[r][cc] = 2
                to_process.append((r, cc))
            elif result[r][cc] == 2:
                break
            elif result[r][cc] == 5:
                break

    return result
