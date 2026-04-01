def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has regions separated by zero rows, each region has 3 rows
    2. Each region divided into sections (every 4th column is separator)
    3. For 3 regions: return first region unchanged
    4. For 2 regions: merge by selecting best-fit rows from both regions
    5. For 1 region: transpose each 3x3 section

    Procedure:
    1. Extract all 3-row regions
    2. Apply transformation based on region count
    3. For 1 region: transpose sections
    4. For 2 regions: combine using row selection heuristics
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])

    # Extract regions
    regions = []
    i = 0
    while i < rows:
        if all(grid[i][j] == 0 for j in range(cols)):
            i += 1
            continue

        region_start = i
        while i < rows and not all(grid[i][j] == 0 for j in range(cols)):
            i += 1

        if i - region_start == 3:
            regions.append([grid[r][:] for r in range(region_start, i)])

    if not regions:
        return [[0] * cols for _ in range(3)]

    if len(regions) == 3:
        return regions[0]

    elif len(regions) == 2:
        # Merge two regions
        region1, region2 = regions
        result = [[0] * cols for _ in range(3)]

        for c in range(0, cols, 4):
            if c + 3 <= cols:
                # Get all 6 rows from both regions for this section
                rows_r1 = [region1[r][c:c+3] for r in range(3)]
                rows_r2 = [region2[r][c:c+3] for r in range(3)]

                # Build output by selecting rows based on properties
                for out_r in range(3):
                    # Try to match patterns: select row that best fits output position
                    candidates = rows_r1 + rows_r2
                    best_row = None
                    best_score = -999

                    for idx, candidate in enumerate(candidates):
                        score = 0

                        # Prefer matching row indices (R1[out_r] or R2[out_r])
                        if idx == out_r or idx == out_r + 3:
                            score += 100

                        # Row-specific preferences
                        if out_r == 1:  # Middle row
                            # Prefer horizontal solid lines
                            if candidate[0] == candidate[1] == candidate[2] and candidate[0] != 0:
                                score += 50
                            # Prefer rows with all 3 non-zero
                            if all(v != 0 for v in candidate):
                                score += 30
                        else:  # First or last row
                            # Prefer rows with non-zero edges
                            if candidate[0] != 0:
                                score += 15
                            if candidate[2] != 0:
                                score += 15

                        # General: prefer more non-zero values
                        score += sum(1 for v in candidate if v != 0) * 5

                        if score > best_score:
                            best_score = score
                            best_row = candidate

                    if best_row:
                        result[out_r][c:c+3] = best_row[:]

        return result

    else:  # 1 region
        # Transpose each 3x3 section
        region = regions[0]
        result = [[0] * cols for _ in range(3)]

        for c in range(0, cols, 4):
            if c + 3 <= cols:
                # Extract section
                section = [region[r][c:c+3] for r in range(3)]

                # Transpose: (r, c) -> (c, r)
                for r in range(3):
                    for col in range(3):
                        result[r][c+col] = section[col][r]

        return result
