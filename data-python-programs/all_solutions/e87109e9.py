def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Crop to rows 6-24 (remove top legend section)
    2. Find 8s marker (small 2x2 block) - this is the anchor
    3. Draw 3 vertical lines (2-wide each): left, center (at 8s), right
    4. Draw horizontal lines (2-wide) at 8s row and block boundaries
    5. Lines are segmented - they don't always span full width/height
    6. Vertical line positions: (leftmost_block_right + 1), (8s_col), (rightmost_block_left - 2)
    7. Each line segment has specific extent based on surrounding blocks

    Procedure:
    1. Crop and find background, 8s, blocks
    2. Calculate exact 3 vertical line positions
    3. Calculate horizontal line positions
    4. Draw vertical lines with proper segmentation
    5. Draw horizontal lines with proper extents
    """
    from collections import Counter

    # Crop to main section
    result = [row[:] for row in grid[6:25]]
    height, width = len(result), len(result[0])

    # Find background
    counts = Counter()
    for row in result:
        for val in row:
            if val != 8:
                counts[val] += 1
    bg = max(counts, key=counts.get)

    # Find 8s position
    eights = [(r, c) for r in range(height) for c in range(width) if result[r][c] == 8]
    if not eights:
        return result

    er_min = min(r for r, c in eights)
    er_max = max(r for r, c in eights)
    ec_min = min(c for r, c in eights)
    ec_max = max(c for r, c in eights)

    # Find all blocks
    blocks = {}
    for r in range(height):
        for c in range(width):
            if result[r][c] not in [bg, 8]:
                color = result[r][c]
                if color not in blocks:
                    blocks[color] = []
                blocks[color].append((r, c))

    block_bounds = []
    for color, cells in blocks.items():
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        block_bounds.append((min(rs), max(rs), min(cs), max(cs)))

    # Determine 3 vertical line column positions
    vcols = []

    # Center: 8s position
    vcols.append((ec_min, ec_min + 1))

    # Left: rightmost edge of leftmost block + 1
    left_blocks = [(c_min, c_max) for r_min, r_max, c_min, c_max in block_bounds if c_max < ec_min]
    if left_blocks:
        rightmost_left = max(c_max for c_min, c_max in left_blocks)
        vcols.append((rightmost_left + 1, rightmost_left + 2))

    # Right: leftmost edge of rightmost block - 2
    right_blocks = [(c_min, c_max) for r_min, r_max, c_min, c_max in block_bounds if c_min > ec_max]
    if right_blocks:
        leftmost_right = min(c_min for c_min, c_max in right_blocks)
        vcols.append((leftmost_right - 2, leftmost_right - 1))

    # Sort vertical lines by column
    vcols = sorted(vcols)

    # Determine horizontal line row positions
    hrows = [(er_min, er_min + 1)]  # Always include 8s row

    # Find horizontal lines at block boundaries
    for r_min, r_max, c_min, c_max in block_bounds:
        if r_max < er_min - 1:  # Block above 8s
            hrows.append((r_max + 1, r_max + 2))
        elif r_min > er_max + 1:  # Block below 8s
            hrows.append((r_min - 2, r_min - 1))

    hrows = sorted(set(hrows))

    # Draw vertical lines with segmentation by blocks and horizontal lines
    for vc1, vc2 in vcols:
        # Determine vertical extent for this line
        if (vc1, vc2) == (ec_min, ec_min + 1):
            # Center line (8s position): extends full height with block cutouts
            r_start, r_end = 0, height - 1

            # Find blocks that intersect this column
            for r_min, r_max, c_min, c_max in block_bounds:
                if c_min <= vc1 <= c_max or c_min <= vc2 <= c_max:
                    if r_max < er_min:
                        r_start = max(r_start, r_max + 1)
                    elif r_min > er_max:
                        r_end = min(r_end, r_min - 1)

            for r in range(r_start, r_end + 1):
                if result[r][vc1] == bg:
                    result[r][vc1] = 8
                if result[r][vc2] == bg:
                    result[r][vc2] = 8

        elif vc1 < ec_min:
            # Left line: starts from 8s row, extends downward
            r_start = er_min
            r_end = height - 1

            # Check for blocks that would cut it short
            for r_min, r_max, c_min, c_max in block_bounds:
                if c_min <= vc1 <= c_max or c_min <= vc2 <= c_max:
                    if r_min > er_max:
                        r_end = min(r_end, r_min - 1)

            for r in range(r_start, r_end + 1):
                if result[r][vc1] == bg:
                    result[r][vc1] = 8
                if result[r][vc2] == bg:
                    result[r][vc2] = 8

        else:
            # Right line: starts from top, extends to 8s row
            r_start = 0
            r_end = er_min - 1

            # Check for blocks that would cut it short
            for r_min, r_max, c_min, c_max in block_bounds:
                if c_min <= vc1 <= c_max or c_min <= vc2 <= c_max:
                    if r_max < er_min:
                        r_start = max(r_start, r_max + 1)

            for r in range(r_start, r_end + 1):
                if result[r][vc1] == bg:
                    result[r][vc1] = 8
                if result[r][vc2] == bg:
                    result[r][vc2] = 8

    # Draw horizontal lines with varying column extents
    for hr1, hr2 in hrows:
        # Determine horizontal extent
        c_left = min(vc1 for vc1, vc2 in vcols)
        c_right = max(vc2 for vc1, vc2 in vcols) - 1

        # Bottom-most horizontal line extends further right
        is_bottom = all(hr1 >= other_hr1 for other_hr1, other_hr2 in hrows)

        # Extend to blocks that intersect this row
        for r_min, r_max, c_min, c_max in block_bounds:
            if r_min <= hr2 and r_max >= hr1:
                # Block intersects this horizontal line
                if is_bottom and c_min > ec_max:
                    c_right = max(c_right, c_max + 1)
                elif (hr1, hr2) != (er_min, er_min + 1) and c_min > ec_max:
                    c_right = max(c_right, c_max + 1)

        for c in range(c_left, min(c_right + 1, width)):
            if 0 <= hr1 < height and result[hr1][c] == bg:
                result[hr1][c] = 8
            if 0 <= hr2 < height and result[hr2][c] == bg:
                result[hr2][c] = 8

    return result
