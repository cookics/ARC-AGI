def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a cluster of 4s forming a pattern with 6s
    2. Find the bounding box of all 4s
    3. Extract that rectangular region
    4. Try multiple transformations (rotations, flips)
    5. Search for locations where the 6s pattern matches (subset matching)
    6. Where input has 6, keep 6; where pattern has 4, output 4

    Procedure:
    1. Find all 4s and compute bounding box
    2. Extract the rectangular region
    3. Try all transformations
    4. For each transformation, find matching locations
    5. Fill appropriately
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all 4s
    fours = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 4:
                fours.append((i, j))

    if not fours:
        return result

    # Get bounding box
    min_r = min(r for r, c in fours)
    max_r = max(r for r, c in fours)
    min_c = min(c for r, c in fours)
    max_c = max(c for r, c in fours)

    # Extract region
    region = []
    for i in range(min_r, max_r + 1):
        row = []
        for j in range(min_c, max_c + 1):
            row.append(grid[i][j])
        region.append(row)

    def rotate_90(mat):
        h, w = len(mat), len(mat[0])
        result = [[0] * h for _ in range(w)]
        for i in range(h):
            for j in range(w):
                result[j][h - 1 - i] = mat[i][j]
        return result

    def flip_h(mat):
        return [row[::-1] for row in mat]

    def flip_v(mat):
        return mat[::-1]

    # Generate all transformations
    transformations = [region]

    # Rotations
    r90 = rotate_90(region)
    r180 = rotate_90(r90)
    r270 = rotate_90(r180)
    transformations.extend([r90, r180, r270])

    # Flips
    transformations.append(flip_h(region))
    transformations.append(flip_v(region))

    # Combinations
    transformations.append(flip_h(r90))
    transformations.append(flip_v(r90))

    # Try each transformation
    for transformed in transformations:
        t_h, t_w = len(transformed), len(transformed[0])

        # Extract 6s pattern
        six_pattern = set()
        for i in range(t_h):
            for j in range(t_w):
                if transformed[i][j] == 6:
                    six_pattern.add((i, j))

        # Search for matching locations
        for start_r in range(rows):
            for start_c in range(cols):
                # Calculate how much of pattern is within bounds
                in_bounds_cells = 0
                total_cells = t_h * t_w
                for i in range(t_h):
                    for j in range(t_w):
                        if start_r + i < rows and start_c + j < cols:
                            in_bounds_cells += 1

                # Skip if less than 60% of pattern is in bounds (too partial)
                if in_bounds_cells < total_cells * 0.6:
                    continue

                # Skip if overlaps with original region
                if not (start_r > max_r or start_r + t_h - 1 < min_r or
                        start_c > max_c or start_c + t_w - 1 < min_c):
                    continue

                # Check if pattern 6s match input 6s
                match = True
                in_bounds_sixes = 0
                for dr, dc in six_pattern:
                    r, c = start_r + dr, start_c + dc
                    if r >= rows or c >= cols:
                        # Out of bounds is only OK if we have mostly in-bounds pattern
                        if in_bounds_cells >= total_cells * 0.9:
                            continue
                        else:
                            match = False
                            break
                    in_bounds_sixes += 1
                    if grid[r][c] != 6:
                        match = False
                        break

                # Require at least some in-bounds 6s to match (avoid trivial empty patterns)
                if match and len(six_pattern) > 0 and in_bounds_sixes == 0:
                    match = False

                if match:
                    # Apply transformation: only fill 4s where pattern has 4 and input doesn't have 6
                    for i in range(t_h):
                        for j in range(t_w):
                            r, c = start_r + i, start_c + j
                            if r >= rows or c >= cols:
                                continue  # Skip out-of-bounds cells
                            if grid[r][c] != 6 and transformed[i][j] == 4:
                                result[r][c] = 4

    return result
