def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Two 3x3 cross patterns made of 2s with 5 at center
    2. Each cross has one missing cell (0) creating a gap
    3. A thick corridor of 4s connects the two patterns through their gaps
    4. Corridor structure: gap -> extension -> connector -> main band -> connector -> extension -> gap
    5. Main band spans horizontally between the patterns (avoiding their footprints)
    6. Connectors link gaps to the main band

    Procedure:
    1. Find patterns, gaps, and gap directions
    2. Compute gap extensions (1 step in gap direction)
    3. Determine main band row range (between patterns)
    4. Determine main band column range (spanning across)
    5. Fill main band, then fill connectors from gaps to main band
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find pattern centers (cells with value 5)
    fives = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if len(fives) != 2:
        return result

    # Analyze each pattern
    def analyze_pattern(center_r, center_c):
        bounds = (center_r - 1, center_r + 1, center_c - 1, center_c + 1)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            gr, gc = center_r + dr, center_c + dc
            if 0 <= gr < rows and 0 <= gc < cols and grid[gr][gc] == 0:
                or_, oc = center_r - dr, center_c - dc
                if 0 <= or_ < rows and 0 <= oc < cols and grid[or_][oc] == 2:
                    return bounds, (gr, gc), (dr, dc)
        return None, None, None

    bounds1, gap1, dir1 = analyze_pattern(*fives[0])
    bounds2, gap2, dir2 = analyze_pattern(*fives[1])
    if not bounds1 or not bounds2:
        return result

    # Sort by row position
    if bounds1[0] < bounds2[0]:
        upper_b, upper_g, upper_d = bounds1, gap1, dir1
        lower_b, lower_g, lower_d = bounds2, gap2, dir2
    else:
        upper_b, upper_g, upper_d = bounds2, gap2, dir2
        lower_b, lower_g, lower_d = bounds1, gap1, dir1

    # Main band rows: between the two patterns
    band_r_min = upper_b[1] + 1
    band_r_max = lower_b[0] - 1

    # Main band columns: determine based on gaps and extensions
    # Use gap columns and adjust based on directions
    gap_cols = [upper_g[1], lower_g[1]]
    ext_cols = [upper_g[1] + upper_d[1], lower_g[1] + lower_d[1]]

    print(f"DEBUG: upper_g={upper_g}, upper_d={upper_d}, lower_g={lower_g}, lower_d={lower_d}")
    print(f"DEBUG: gap_cols={gap_cols}, ext_cols={ext_cols}")

    # Band columns span from min to max, but adjust for vertical gaps
    band_c_min = min(gap_cols + ext_cols)
    band_c_max = max(gap_cols + ext_cols)
    print(f"DEBUG: initial band_c_min={band_c_min}, band_c_max={band_c_max}")

    # Adjust: if a gap opens vertically, its column + adjacent should be in band
    if upper_d[1] == 0:  # vertical gap
        if upper_g[1] == band_c_min:
            band_c_min += 1
            print(f"DEBUG: Adjusted band_c_min for upper gap: {band_c_min}")
    if lower_d[1] == 0:  # vertical gap
        if lower_g[1] == band_c_min:
            band_c_min += 1
            print(f"DEBUG: Adjusted band_c_min for lower gap: {band_c_min}")

    print(f"DEBUG: band rows {band_r_min}-{band_r_max}, cols {band_c_min}-{band_c_max}")

    # Fill main band
    if band_r_min <= band_r_max:
        for r in range(band_r_min, band_r_max + 1):
            for c in range(band_c_min, band_c_max + 1):
                if grid[r][c] == 0:
                    result[r][c] = 4
                    if r == 6 and c == 2:
                        print(f"DEBUG: Filling row 6 col 2")

    # Fill upper connector
    ur, uc = upper_g
    ext_ur, ext_uc = ur + upper_d[0], uc + upper_d[1]

    # Always fill gap
    if grid[ur][uc] == 0:
        result[ur][uc] = 4

    # Fill extension only if outside main band
    if 0 <= ext_ur < rows and 0 <= ext_uc < cols and grid[ext_ur][ext_uc] == 0:
        if ext_ur < band_r_min or ext_ur > band_r_max:
            result[ext_ur][ext_uc] = 4

    # Connector from extension/gap to main band
    if upper_d[0] == 0:  # horizontal gap: vertical connector needed
        conn_c = ext_uc
        start_r = ext_ur + 1 if ext_ur < band_r_min else ext_ur
        for r in range(start_r, band_r_min):
            if 0 <= r < rows and 0 <= conn_c < cols and grid[r][conn_c] == 0:
                result[r][conn_c] = 4
    else:  # vertical gap: may need horizontal connector
        if uc < band_c_min:
            for c in range(uc + 1, band_c_min):
                if 0 <= ur < rows and grid[ur][c] == 0:
                    result[ur][c] = 4
        elif uc > band_c_max:
            for c in range(band_c_max + 1, uc):
                if 0 <= ur < rows and grid[ur][c] == 0:
                    result[ur][c] = 4
        # Vertical connector from extension to band
        start_r = ext_ur + 1 if ext_ur < band_r_min else ext_ur
        for r in range(start_r, band_r_min):
            if 0 <= r < rows and 0 <= ext_uc < cols and grid[r][ext_uc] == 0:
                result[r][ext_uc] = 4

    # Fill lower connector
    lr, lc = lower_g

    # Always fill gap
    if grid[lr][lc] == 0:
        result[lr][lc] = 4

    # Connector from main band to gap
    if lower_d[0] == 0:  # horizontal gap: vertical connector needed
        ext_lr, ext_lc = lr + lower_d[0], lc + lower_d[1]
        # Fill extension if outside main band
        if 0 <= ext_lr < rows and 0 <= ext_lc < cols and grid[ext_lr][ext_lc] == 0:
            if ext_lr < band_r_min or ext_lr > band_r_max:
                result[ext_lr][ext_lc] = 4
        # Vertical connector
        conn_c = ext_lc
        for r in range(band_r_max + 1, lr):
            if 0 <= r < rows and 0 <= conn_c < cols and grid[r][conn_c] == 0:
                result[r][conn_c] = 4
    else:  # vertical gap: horizontal connector to main band edge
        if lc < band_c_min:
            for c in range(lc + 1, band_c_min):
                if 0 <= lr < rows and grid[lr][c] == 0:
                    result[lr][c] = 4
        elif lc > band_c_max:
            for c in range(band_c_max + 1, lc):
                if 0 <= lr < rows and grid[lr][c] == 0:
                    result[lr][c] = 4

    return result
