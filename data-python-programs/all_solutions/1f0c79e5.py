def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 9x9 grid with a 2x2 block containing values 2 and another color
    2. Output creates diagonal stripe(s) of width 3 using the non-2 color
    3. The position of value 2 in the 2x2 block determines the diagonal pattern:
       - 2 at NE and SW (anti-diagonal): full / diagonal across grid
       - 2 only at NE: / diagonal from top to bottom of 2x2 block
       - 2 only at SE: \ diagonal from top of 2x2 to bottom
       - 2 everywhere except SE (three 2s): / diagonal with toroidal wrapping

    Procedure:
    1. Find the 2x2 block containing all non-zero values
    2. Identify the drawing color (non-2 value)
    3. Determine pattern based on position of 2s
    4. Draw diagonal stripe of width 3 with appropriate wrapping
    """
    n = len(grid)
    result = [[0] * n for _ in range(n)]

    # Find the 2x2 block (all four cells must be non-zero)
    block_r, block_c = -1, -1
    for r in range(n - 1):
        for c in range(n - 1):
            if (grid[r][c] != 0 and grid[r][c+1] != 0 and
                grid[r+1][c] != 0 and grid[r+1][c+1] != 0):
                block_r, block_c = r, c
                break
        if block_r != -1:
            break

    # Get the 2x2 values
    nw = grid[block_r][block_c]
    ne = grid[block_r][block_c+1]
    sw = grid[block_r+1][block_c]
    se = grid[block_r+1][block_c+1]

    # Find the color to use (non-2, non-0)
    color = 0
    for val in [nw, ne, sw, se]:
        if val != 0 and val != 2:
            color = val
            break

    # Count 2s
    count_2 = [nw, ne, sw, se].count(2)

    center_r = block_r + 0.5
    center_c = block_c + 0.5

    # Determine diagonal type and extent
    if count_2 == 2 and ne == 2 and sw == 2:
        # Two 2s on / diagonal (anti-diagonal) - draw full / diagonal
        sum_val = center_r + center_c
        for r in range(n):
            c_center = sum_val - r
            for c in range(n):
                if abs(c - c_center) < 1.5:
                    result[r][c] = color

    elif count_2 == 1 and ne == 2:
        # One 2 at NE - / diagonal from top to bottom row of 2x2 block
        sum_val = center_r + center_c
        for r in range(block_r + 1):
            c_center = sum_val - r
            for c in range(n):
                if abs(c - c_center) < 1.5:
                    result[r][c] = color
        # Preserve the bottom row of the 2x2
        for c in range(n):
            if grid[block_r + 1][c] != 0:
                result[block_r + 1][c] = color

    elif count_2 == 1 and se == 2:
        # One 2 at SE - \ diagonal from top of 2x2 downward
        diff_val = center_r - center_c
        # Preserve the first row of the 2x2
        for c in range(n):
            if grid[block_r][c] != 0:
                result[block_r][c] = color
        # Draw diagonal from second row onwards
        for r in range(block_r + 1, n):
            c_center = r - diff_val
            for c in range(n):
                if abs(c - c_center) < 1.5:
                    result[r][c] = color

    elif count_2 == 3 and se != 2:
        # Three 2s with non-2 at SE - / diagonal with toroidal wrapping
        sum_val = center_r + center_c
        for r in range(n):
            c_center = sum_val - r
            for c in range(n):
                # Standard width-3 band around c_center
                if abs(c - c_center) < 1.5:
                    result[r][c] = color
                # Toroidal wrapping for left side when diagonal is near right edge
                # Include left-side cells based on how far c_center is from the edge
                elif c_center >= 4 and c <= 8.5 - c_center and c >= max(0, 6 - int(c_center)):
                    result[r][c] = color

    elif count_2 == 3 and sw != 2:
        # Three 2s with non-2 at SW - \ diagonal with wrapping
        diff_val = center_r - center_c
        for r in range(n):
            c_center = r - diff_val
            for c in range(n):
                if abs(c - c_center) < 1.5:
                    result[r][c] = color
                elif c_center >= 4 and c <= 8.5 - c_center and c >= max(0, 6 - int(c_center)):
                    result[r][c] = color

    return result
