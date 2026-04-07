def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing integers, with 0 representing empty cells and 4 representing divider lines.
    2. The grid is divided into regions by complete horizontal rows of 4s and/or complete vertical columns of 4s.
    3. Some regions contain patterns of non-zero, non-4 numbers while other regions are mostly empty.
    4. Output should replicate the pattern from the source region to all regions of the same dimensions.

    Procedure:
    1. Identify horizontal divider lines by finding complete rows filled with 4s.
    2. Identify vertical divider lines by finding complete columns filled with 4s.
    3. Create rectangular regions bounded by these divider lines.
    4. Count the non-zero, non-4 content in each region to identify the source region.
    5. Copy the pattern from the source region to all other regions of matching dimensions.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find horizontal dividers
    h_div = None
    for r in range(rows):
        if all(grid[r][c] == 4 for c in range(cols)):
            h_div = r
            break

    # Find vertical dividers
    v_divs = []
    for c in range(cols):
        if all(grid[r][c] == 4 for r in range(rows)):
            v_divs.append(c)

    # Simple case: only horizontal divider
    if h_div is not None and not v_divs:
        top_region = (0, h_div, 0, cols)
        bottom_region = (h_div + 1, rows, 0, cols)

        # Count content in each region
        top_content = 0
        for r in range(0, h_div):
            for c in range(cols):
                if grid[r][c] != 0 and grid[r][c] != 4:
                    top_content += 1

        bottom_content = 0
        for r in range(h_div + 1, rows):
            for c in range(cols):
                if grid[r][c] != 0 and grid[r][c] != 4:
                    bottom_content += 1

        # Copy from region with more content to the other
        if top_content > bottom_content:
            # Copy top to bottom
            for r in range(h_div):
                for c in range(cols):
                    result[h_div + 1 + r][c] = grid[r][c]
        elif bottom_content > top_content:
            # Copy bottom to top
            for r in range(h_div):
                for c in range(cols):
                    result[r][c] = grid[h_div + 1 + r][c]

    # Case with both horizontal and vertical dividers
    elif h_div is not None and v_divs:
        # Create regions
        regions = []

        # Top regions
        prev_c = 0
        for v_c in v_divs + [cols]:
            if v_c - prev_c > 1:  # Skip single-column regions
                regions.append((0, h_div, prev_c, v_c))
            prev_c = v_c + 1

        # Bottom regions
        prev_c = 0
        for v_c in v_divs + [cols]:
            if v_c - prev_c > 1:  # Skip single-column regions
                regions.append((h_div + 1, rows, prev_c, v_c))
            prev_c = v_c + 1

        # Find source region with most content
        source = None
        max_content = 0

        for r1, r2, c1, c2 in regions:
            content = 0
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if grid[r][c] != 0 and grid[r][c] != 4:
                        content += 1

            if content > max_content:
                max_content = content
                source = (r1, r2, c1, c2)

        # Copy source to all regions of same size
        if source and max_content > 0:
            src_r1, src_r2, src_c1, src_c2 = source
            src_h = src_r2 - src_r1
            src_w = src_c2 - src_c1

            for r1, r2, c1, c2 in regions:
                if (r2 - r1) == src_h and (c2 - c1) == src_w:
                    for dr in range(src_h):
                        for dc in range(src_w):
                            result[r1 + dr][c1 + dc] = grid[src_r1 + dr][src_c1 + dc]

    return result
