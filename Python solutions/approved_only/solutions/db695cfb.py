def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has background values, with some 1s and 6s scattered
    2. For pairs of 1s on the same diagonal line, draw connecting line with 1s
    3. For each 6 on a connecting line, extend perpendicular diagonal with 6s
    4. Don't overwrite existing 1s or 6s

    Procedure:
    1. Find all 1s in the grid
    2. For each pair of 1s on same diagonal, draw connecting line
    3. Track which diagonal lines are drawn
    4. For each 6 on these lines, extend perpendicular diagonal
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all 1s
    ones = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 1]

    # Track connecting lines: backslash (row-col=const) and slash (row+col=const)
    backslash_lines = []  # row - col = constant
    slash_lines = []      # row + col = constant

    # Find pairs of 1s on same diagonal and draw connecting lines
    for i in range(len(ones)):
        for j in range(i + 1, len(ones)):
            r1, c1 = ones[i]
            r2, c2 = ones[j]

            # Check if on same \ diagonal (row - col = constant)
            if r1 - c1 == r2 - c2:
                diff = r1 - c1
                backslash_lines.append(diff)
                # Draw line
                if r1 > r2:
                    r1, c1, r2, c2 = r2, c2, r1, c1
                for k in range(r2 - r1 + 1):
                    r, c = r1 + k, c1 + k
                    if result[r][c] not in [1, 6]:
                        result[r][c] = 1

            # Check if on same / diagonal (row + col = constant)
            elif r1 + c1 == r2 + c2:
                total = r1 + c1
                slash_lines.append(total)
                # Draw line
                if r1 > r2:
                    r1, c1, r2, c2 = r2, c2, r1, c1
                for k in range(r2 - r1 + 1):
                    r, c = r1 + k, c1 - k
                    if result[r][c] not in [1, 6]:
                        result[r][c] = 1

    # For each 6, check if on a connecting line and extend perpendicular
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6:
                # Check if on a \ connecting line
                if r - c in backslash_lines:
                    # Extend / diagonal (row + col = constant)
                    target_sum = r + c
                    for rr in range(rows):
                        cc = target_sum - rr
                        if 0 <= cc < cols:
                            if result[rr][cc] not in [1, 6]:
                                result[rr][cc] = 6

                # Check if on a / connecting line
                if r + c in slash_lines:
                    # Extend \ diagonal (row - col = constant)
                    target_diff = r - c
                    for rr in range(rows):
                        cc = rr - target_diff
                        if 0 <= cc < cols:
                            if result[rr][cc] not in [1, 6]:
                                result[rr][cc] = 6

    return result
