def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a small pattern in top-left corner (rest is background 7)
    2. Pattern has cross-shaped separator dividing into quadrants
    3. Output tiles using diagonal shift: output[i][j] = pattern_1d[(j - i) % period]
    4. 1D pattern created by reflecting first row around separator

    Procedure:
    1. Find pattern region and separator column
    2. Reflect first row to create 1D pattern
    3. Apply diagonal shift tiling with wraparound
    """

    rows, cols = len(grid), len(grid[0])
    background = 7

    # Find pattern bounding box
    max_r, max_c = 0, 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                max_r, max_c = max(max_r, r), max(max_c, c)

    pattern_h, pattern_w = max_r + 1, max_c + 1

    # Find separator column
    sep_col = -1
    for c in range(pattern_w):
        vals = set(grid[r][c] for r in range(pattern_h))
        if len(vals) == 1 and list(vals)[0] != background:
            sep_col = c
            break

    # Create 1D pattern from first row with reflection
    # For quadrant width w, period is 2*w: [quadrant][separator][reflected]
    quad_w = sep_col
    period = 2 * quad_w

    pattern_1d = []
    # Add quadrant part
    for c in range(quad_w):
        pattern_1d.append(grid[0][c])
    # Add separator
    pattern_1d.append(grid[0][sep_col])
    # Add reflection of quadrant (reverse order, excluding last which would duplicate separator neighbor)
    for c in range(quad_w - 1, -1, -1):
        pattern_1d.append(grid[0][c])

    # Only use first 'period' elements
    pattern_1d = pattern_1d[:period]

    # Diagonal shift tiling
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            idx = (j - i) % period
            row.append(pattern_1d[idx])
        result.append(row)

    return result
