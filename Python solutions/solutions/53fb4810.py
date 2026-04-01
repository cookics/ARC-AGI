def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Cross patterns made of 1s are preserved (anchors)
    2. Isolated pairs (vertical or horizontal) of non-8, non-1 values get extended vertically
    3. Existing long patterns should not be further extended
    4. An isolated pair is not part of a longer existing sequence

    Procedure:
    1. Find isolated vertical pairs and extend them vertically
    2. Find isolated horizontal pairs and extend them vertically at their columns
    3. Stop extension when hitting 1s or boundaries
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find isolated vertical pairs
    for c in range(cols):
        for r in range(rows - 1):
            val1, val2 = grid[r][c], grid[r + 1][c]

            # Skip if either value is background or cross or same
            if val1 in [8, 1] or val2 in [8, 1] or val1 == val2:
                continue

            # Check if this vertical pair is truly isolated
            isolated = True

            # Check if there's continuation above or below (longer pattern)
            if r > 0 and grid[r - 1][c] not in [8, 1]:
                isolated = False
            if r + 2 < rows and grid[r + 2][c] not in [8, 1]:
                isolated = False

            # Check surrounding area for other non-8, non-1 values
            for dr in range(-1, 3):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if dc != 0 and grid[nr][nc] not in [8, 1]:
                            isolated = False
                            break
                if not isolated:
                    break

            if isolated:
                # Extend vertical pattern in both directions
                # Extend upward
                for up_r in range(r - 1, -1, -1):
                    if grid[up_r][c] == 1:  # Stop at 1s
                        break
                    steps_up = r - up_r
                    if steps_up % 2 == 1:
                        result[up_r][c] = val2
                    else:
                        result[up_r][c] = val1

                # Extend downward
                for down_r in range(r + 2, rows):
                    if grid[down_r][c] == 1:  # Stop at 1s
                        break
                    steps_down = down_r - r
                    if steps_down % 2 == 0:
                        result[down_r][c] = val1
                    else:
                        result[down_r][c] = val2

                break  # Only process one isolated pair per column

    # Find isolated horizontal pairs
    for r in range(rows):
        for c in range(cols - 1):
            val1, val2 = grid[r][c], grid[r][c + 1]

            # Skip if either value is background or cross or same
            if val1 in [8, 1] or val2 in [8, 1] or val1 == val2:
                continue

            # Check if this horizontal pair is truly isolated
            isolated = True

            # Check if there's continuation left or right (longer pattern)
            if c > 0 and grid[r][c - 1] not in [8, 1]:
                isolated = False
            if c + 2 < cols and grid[r][c + 2] not in [8, 1]:
                isolated = False

            # Check surrounding area for other non-8, non-1 values (but allow 2x2 blocks)
            for dr in range(-1, 2):
                for dc in range(-1, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if dr != 0 and grid[nr][nc] not in [8, 1]:
                            # Special case: allow if it forms a 2x2 block pattern
                            if not (dr == 1 and dc in [0, 1] and r + 1 < rows):
                                isolated = False
                                break
                if not isolated:
                    break

            if isolated:
                # Extend horizontal pair vertically at both columns
                # Extend upward
                for up_r in range(r - 1, -1, -1):
                    if grid[up_r][c] == 1 or grid[up_r][c + 1] == 1:  # Stop at 1s
                        break
                    result[up_r][c] = val1
                    result[up_r][c + 1] = val2

                # Extend downward
                for down_r in range(r + 1, rows):
                    if grid[down_r][c] == 1 or grid[down_r][c + 1] == 1:  # Stop at 1s
                        break
                    result[down_r][c] = val1
                    result[down_r][c + 1] = val2

                break  # Only process one isolated pair per row

    # Find isolated 2x2 blocks
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Check if we have a 2x2 block of non-8, non-1 values
            block = [[grid[r][c], grid[r][c + 1]], [grid[r + 1][c], grid[r + 1][c + 1]]]

            # Skip if any value is background or cross
            if any(val in [8, 1] for row in block for val in row):
                continue

            # Check if this 2x2 block is isolated
            isolated = True

            # Check surrounding area
            for dr in range(-1, 3):
                for dc in range(-1, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        # Skip the 2x2 block itself
                        if 0 <= dr <= 1 and 0 <= dc <= 1:
                            continue
                        if grid[nr][nc] not in [8, 1]:
                            isolated = False
                            break
                if not isolated:
                    break

            if isolated:
                # Extend 2x2 block pattern vertically
                # Extend upward
                for up_r in range(r - 2, -1, -2):  # Step by 2 to repeat the 2x2 pattern
                    if (
                        grid[up_r][c] == 1
                        or grid[up_r][c + 1] == 1
                        or grid[up_r + 1][c] == 1
                        or grid[up_r + 1][c + 1] == 1
                    ):
                        break
                    result[up_r][c] = block[0][0]
                    result[up_r][c + 1] = block[0][1]
                    result[up_r + 1][c] = block[1][0]
                    result[up_r + 1][c + 1] = block[1][1]

                # Extend downward
                for down_r in range(
                    r + 2, rows - 1, 2
                ):  # Step by 2 to repeat the 2x2 pattern
                    if (
                        grid[down_r][c] == 1
                        or grid[down_r][c + 1] == 1
                        or grid[down_r + 1][c] == 1
                        or grid[down_r + 1][c + 1] == 1
                    ):
                        break
                    result[down_r][c] = block[0][0]
                    result[down_r][c + 1] = block[0][1]
                    result[down_r + 1][c] = block[1][0]
                    result[down_r + 1][c + 1] = block[1][1]

                break  # Only process one isolated block per position

    return result
