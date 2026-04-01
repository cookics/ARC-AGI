def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a background color and several colored shapes
    2. Output makes each shape both vertically AND horizontally symmetric
    3. Vertical symmetry: mirror rows around horizontal center of bounding box
    4. Horizontal symmetry: mirror columns within each row, extending bounding box if width is even

    Procedure:
    1. Find background color (most frequent)
    2. For each non-background color:
       a. Apply vertical mirroring across all rows in bounding box
       b. Apply horizontal mirroring row by row, extending bbox if width is even
    3. Return modified grid
    """

    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most common)
    color_counts = Counter(grid[r][c] for r in range(rows) for c in range(cols))
    background = max(color_counts, key=color_counts.get)

    # Get all non-background colors
    colors = set(color_counts.keys()) - {background}

    # Create output grid
    result = [row[:] for row in grid]

    # Make each color symmetric
    for color in colors:
        # Find all cells with this color
        cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]

        if not cells:
            continue

        # Find bounding box
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Step 1: Apply vertical mirroring
        for i in range(min_r, max_r + 1):
            paired_row = min_r + max_r - i
            for j in range(min_c, max_c + 1):
                if grid[i][j] == color or grid[paired_row][j] == color:
                    result[i][j] = color
                    result[paired_row][j] = color

        # Step 2: Apply horizontal mirroring per row
        for row in range(min_r, max_r + 1):
            # Find columns with color in this row (after vertical mirroring)
            row_cols = [c for c in range(cols) if result[row][c] == color]

            if len(row_cols) < 2:
                continue

            # Get bounding box for this row
            row_min_c = min(row_cols)
            row_max_c = max(row_cols)
            width = row_max_c - row_min_c + 1

            # Check if there's a gap (discontinuity) in the pattern
            num_gaps = width - len(row_cols)

            # Only apply horizontal mirroring if:
            # - width is even AND
            # - there's exactly ONE gap cell (which will be at the center)
            if width % 2 == 0 and num_gaps == 1:
                row_max_c += 1
                width += 1

                # Apply horizontal mirroring
                for j in range(width):
                    left_col = row_min_c + j
                    right_col = row_max_c - j

                    if left_col >= right_col or left_col < 0 or right_col >= cols:
                        break

                    # If either side has the color, both get it
                    if result[row][left_col] == color or result[row][right_col] == color:
                        result[row][left_col] = color
                        result[row][right_col] = color

    return result
