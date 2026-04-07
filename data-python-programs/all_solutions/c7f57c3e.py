def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has cross patterns made with 1s (frame) and 2 (center)
    2. Each cross has extensions below/above with other values
    3. Similar cross patterns swap their extension content
    4. Extensions are values below the cross (not 1, not core 2, not background)
    5. Larger patterns also transform based on which extension pattern "wins"

    Procedure:
    1. Find all small cross patterns (1s forming cross with 2 in center)
    2. For each cross, extract extension values below it
    3. Identify pairs of crosses with same structure
    4. Swap extension values between pairs
    5. Apply the "winning" extension to all other patterns
    """

    from collections import Counter

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find background
    flat = [cell for row in grid for cell in row]
    background = Counter(flat).most_common(1)[0][0]

    # Find small cross patterns: looking for structure like
    #   1
    # 1 2 1
    # Then check below for extensions

    crosses = []

    for r in range(rows - 3):
        for c in range(1, cols - 1):
            # Check for cross pattern
            if (grid[r][c] == background and grid[r][c+1] == 1 and grid[r][c+2] == background and
                grid[r+1][c] == 1 and grid[r+1][c+1] == 2 and grid[r+1][c+2] == 1):

                # Found a cross at (r, c) to (r+1, c+2)
                # Check for extensions below (rows r+2 and r+3)
                ext_vals = []

                if r + 3 < rows:
                    # Check row r+2, col c+1
                    val1 = grid[r+2][c+1]
                    # Check row r+3, cols c to c+2
                    val2_0 = grid[r+3][c]
                    val2_1 = grid[r+3][c+1]
                    val2_2 = grid[r+3][c+2]

                    ext_vals = [val1, val2_0, val2_1, val2_2]

                crosses.append({
                    'r': r,
                    'c': c,
                    'ext_vals': ext_vals
                })

    # If we have exactly 2 crosses, swap their extensions
    if len(crosses) == 2:
        c1, c2 = crosses[0], crosses[1]
        ext1, ext2 = c1['ext_vals'], c2['ext_vals']

        if len(ext1) == 4 and len(ext2) == 4:
            # Swap extensions
            r1, c1_col = c1['r'], c1['c']
            r2, c2_col = c2['r'], c2['c']

            # Track which cells are part of cross extensions (to exclude from global transform)
            cross_ext_cells = set()
            cross_ext_cells.add((r1+2, c1_col+1))
            cross_ext_cells.add((r1+3, c1_col))
            cross_ext_cells.add((r1+3, c1_col+1))
            cross_ext_cells.add((r1+3, c1_col+2))
            cross_ext_cells.add((r2+2, c2_col+1))
            cross_ext_cells.add((r2+3, c2_col))
            cross_ext_cells.add((r2+3, c2_col+1))
            cross_ext_cells.add((r2+3, c2_col+2))

            # Apply ext2 to cross1 position
            result[r1+2][c1_col+1] = ext2[0]
            result[r1+3][c1_col] = ext2[1]
            result[r1+3][c1_col+1] = ext2[2]
            result[r1+3][c1_col+2] = ext2[3]

            # Apply ext1 to cross2 position
            result[r2+2][c2_col+1] = ext1[0]
            result[r2+3][c2_col] = ext1[1]
            result[r2+3][c2_col+1] = ext1[2]
            result[r2+3][c2_col+2] = ext1[3]

            # Also transform all other occurrences of the extension values
            unique1 = set(v for v in ext1 if v != background)
            unique2 = set(v for v in ext2 if v != background)

            # Find which values are unique to each extension
            only_in_1 = unique1 - unique2
            only_in_2 = unique2 - unique1

            # Transform all cells with these unique values
            for r in range(rows):
                for c in range(cols):
                    # Skip cross extension cells (already swapped)
                    if (r, c) in cross_ext_cells:
                        continue

                    val = grid[r][c]

                    # Skip background and 1s
                    if val == background or val == 1:
                        continue

                    # Skip core 2s (2s with multiple adjacent 1s)
                    if val == 2:
                        adj_ones = 0
                        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                if grid[nr][nc] == 1:
                                    adj_ones += 1
                        if adj_ones >= 2:
                            continue

                    # Transform values
                    if val in only_in_1:
                        # Transform to most common value in ext2
                        ext2_vals = [v for v in ext2 if v != background]
                        if ext2_vals:
                            result[r][c] = Counter(ext2_vals).most_common(1)[0][0]
                    elif val in only_in_2:
                        # Transform to most common value in ext1
                        ext1_vals = [v for v in ext1 if v != background]
                        if ext1_vals:
                            result[r][c] = Counter(ext1_vals).most_common(1)[0][0]

    return result
