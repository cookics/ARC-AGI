def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 0s (background), 8s (forming a frame), 7s (interior markers), and exactly two 9s (control points)
    2. The 8s form a rectangular bounding box/frame
    3. Output replaces some 8s with 9s on the perimeter of the frame based on the positions of the two 9s
    4. The two 9s define either a horizontal line (same row), vertical line (same column), or diagonal pattern

    Procedure:
    1. Find the two 9s in the grid
    2. Find the bounding box of all 8s
    3. Determine fill pattern:
       - Same row: horizontal split - fill perimeter from that row downward
       - Same column: vertical split - fill perimeter with diagonal pattern
       - Different row and column: fill three sides (exclude side closest to 9s)
    4. Replace appropriate perimeter 8s with 9s
    """
    import copy
    result = copy.deepcopy(grid)

    # Find the two 9s
    nines = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 9:
                nines.append((i, j))

    # Find bounding box of all 8s
    min_row, max_row = float('inf'), float('-inf')
    min_col, max_col = float('inf'), float('-inf')
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 8:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    r1, c1 = nines[0]
    r2, c2 = nines[1]

    if r1 == r2:  # Horizontal line (same row)
        split_row = r1
        # If 9s are above bbox, fill all; if within, fill from split_row down
        fill_start = split_row if split_row >= min_row else min_row

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 8:
                    # Check if on perimeter
                    is_perimeter = (i == min_row or i == max_row or j == min_col or j == max_col)
                    if is_perimeter and i >= fill_start:
                        result[i][j] = 9

    elif c1 == c2:  # Vertical line (same column) - diagonal fill pattern
        split_col = c1
        mid_row = (min_row + max_row) / 2

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 8:
                    # Top half: fill right portion (cols >= split_col)
                    # At mid row: fill only split column
                    # Bottom half: fill left portion (cols <= split_col)
                    if i < mid_row and j >= split_col:
                        result[i][j] = 9
                    elif i == mid_row and j == split_col:
                        result[i][j] = 9
                    elif i > mid_row and j <= split_col:
                        result[i][j] = 9

    else:  # Diagonal line - different row and column
        # Determine which side has the 9s
        avg_col = (c1 + c2) / 2

        if avg_col >= max_col:  # 9s on the right side
            # Fill all edges except right edge (for middle rows)
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 8:
                        # Fill top and bottom edges completely
                        if i == min_row or i == max_row:
                            result[i][j] = 9
                        # Fill left edge
                        elif j == min_col:
                            result[i][j] = 9
                        # Don't fill right edge for middle rows

    return result
