def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. [0,3,0] brackets mark anchor points
    2. Vertical brackets → horizontal lines extend
    3. Horizontal brackets → vertical lines extend
    4. Transformation: keep bracket cell, next becomes 5, rest become 7
    5. Perpendicular lines from transformed cells are cleared
    6. Original 5s adjacent to transformed cells are removed

    Procedure:
    1. Process horizontal brackets first (priority)
    2. Process vertical brackets
    3. Process standalone long vertical lines
    4. Remove adjacent 5s
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]
    transformed = set()

    # STEP 1: Process horizontal brackets [0,3,0] in rows
    for r in range(rows):
        for c in range(cols - 2):
            if grid[r][c] == 0 and grid[r][c+1] == 3 and grid[r][c+2] == 0:
                bracket_c = c + 1

                # Find vertical extension below bracket
                line = []
                rr = r + 1
                while rr < rows and grid[rr][bracket_c] == 3:
                    line.append((rr, bracket_c))
                    rr += 1

                if len(line) >= 1:
                    # First cell becomes 5
                    result[line[0][0]][line[0][1]] = 5
                    transformed.add((line[0][0], line[0][1]))

                    # Rest become 7
                    for i in range(1, len(line)):
                        result[line[i][0]][line[i][1]] = 7
                        transformed.add((line[i][0], line[i][1]))

                    # Clear horizontal extensions from first cell
                    first_r, first_c = line[0]
                    cc = first_c - 1
                    while cc >= 0 and grid[first_r][cc] == 3:
                        result[first_r][cc] = 7
                        transformed.add((first_r, cc))
                        cc -= 1
                    cc = first_c + 1
                    while cc < cols and grid[first_r][cc] == 3:
                        result[first_r][cc] = 7
                        transformed.add((first_r, cc))
                        cc += 1

    # STEP 2: Process vertical brackets [0,3,0] in columns
    for c in range(cols):
        for r in range(rows - 2):
            if grid[r][c] == 0 and grid[r+1][c] == 3 and grid[r+2][c] == 0:
                bracket_r = r + 1

                # Skip if bracket cell already transformed
                if (bracket_r, c) in transformed:
                    continue

                # Find horizontal extension from bracket
                line = []
                cc = c
                while cc >= 0 and grid[bracket_r][cc] == 3:
                    line.append((bracket_r, cc))
                    cc -= 1
                cc = c + 1
                while cc < cols and grid[bracket_r][cc] == 3:
                    line.append((bracket_r, cc))
                    cc += 1

                line = sorted(line, key=lambda x: x[1])

                if len(line) >= 2:
                    # Skip if any column in the extension already has transformed cells
                    skip = False
                    for _, col in line:
                        for row in range(rows):
                            if (row, col) in transformed:
                                skip = True
                                break
                        if skip:
                            break
                    if skip:
                        continue
                    # Transform: first stays 3, second becomes 5, rest become 7
                    for i, (rr, cc) in enumerate(line):
                        if i == 0:
                            pass  # Keep as 3
                        elif i == 1:
                            result[rr][cc] = 5
                            transformed.add((rr, cc))
                        else:
                            result[rr][cc] = 7
                            transformed.add((rr, cc))

                    # Clear vertical lines from non-first cells
                    for i, (rr, cc) in enumerate(line):
                        if i > 0:
                            r2 = rr - 1
                            while r2 >= 0 and grid[r2][cc] == 3:
                                result[r2][cc] = 7
                                transformed.add((r2, cc))
                                r2 -= 1
                            r2 = rr + 1
                            while r2 < rows and grid[r2][cc] == 3:
                                result[r2][cc] = 7
                                transformed.add((r2, cc))
                                r2 += 1

    # STEP 3: Process standalone long vertical lines
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r][c] == 3 and (r, c) not in transformed:
                line = []
                rr = r
                while rr < rows and grid[rr][c] == 3:
                    line.append((rr, c))
                    rr += 1

                if len(line) >= 4:
                    result[line[0][0]][line[0][1]] = 5
                    transformed.add((line[0][0], line[0][1]))
                    for i in range(1, len(line)):
                        result[line[i][0]][line[i][1]] = 7
                        transformed.add((line[i][0], line[i][1]))

                r = rr
            else:
                r += 1

    # STEP 4: Remove 5s adjacent to transformed cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                adjacent = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) in transformed:
                            adjacent = True
                            break
                    if adjacent:
                        break
                if adjacent:
                    result[r][c] = 7

    return result
