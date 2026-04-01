def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with a background color and cross-shaped patterns
    2. Last row contains horizontal 5-cell bars that encode transformations
    3. Bar color 2 at column C: transform crosses at column C to color 5
    4. Bar color 3 at column C: duplicate crosses at column C upward by (cross_color + 1) steps
    5. Each duplication step is 4 rows upward
    6. Last row is cleared in output

    Procedure:
    1. Identify background color
    2. Find horizontal bars in last row and their center columns
    3. Copy input to result
    4. For bar color 2: transform crosses at that column to color 5
    5. For bar color 3: duplicate crosses upward by (cross_color + 1) * 4 rows
    6. Clear last row
    """
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])
    background = Counter(grid[0]).most_common(1)[0][0]

    # Find horizontal bars (5 consecutive cells of same color) in last row
    bars = []  # [(center_col, bar_color)]
    i = 0
    while i < cols:
        if grid[-1][i] != background:
            color = grid[-1][i]
            start = i
            while i < cols and grid[-1][i] == color:
                i += 1
            length = i - start
            if length == 5:
                center = start + 2
                bars.append((center, color))
        else:
            i += 1

    # Create result as a copy of input
    result = [row[:] for row in grid]

    # Process each bar
    for bar_col, bar_color in bars:
        if bar_color == 2:
            # Transform crosses at this column to color 5
            for r in range(rows - 1):
                # Check if this is part of a cross pattern at bar_col
                if result[r][bar_col] != background:
                    result[r][bar_col] = 5
                # Also handle the horizontal parts of the cross (cols bar_col-2 to bar_col+2)
                for c in range(max(0, bar_col - 2), min(cols, bar_col + 3)):
                    if result[r][c] != background:
                        # Check if this cell is part of a cross centered at bar_col
                        # Cross pattern: row has 5 cells, or row has 3 cells (not middle row)
                        result[r][c] = 5

        elif bar_color == 3:
            # Find the cross color at this column
            cross_color = None
            for r in range(rows - 1):
                if grid[r][bar_col] != background:
                    cross_color = grid[r][bar_col]
                    break

            if cross_color is not None:
                # Duplicate upward by (cross_color + 1) steps, each step is 4 rows
                num_duplications = cross_color + 1

                # Find all rows that have the cross at this column
                cross_rows = []
                for r in range(rows - 1):
                    if grid[r][bar_col] == cross_color:
                        cross_rows.append(r)

                # For each existing cross, duplicate it upward
                for _ in range(num_duplications):
                    new_cross_rows = []
                    for r in cross_rows:
                        new_r = r - 4
                        if new_r >= 0:
                            new_cross_rows.append(new_r)

                    # Copy the cross pattern to the new rows
                    for new_r in new_cross_rows:
                        # Find corresponding source row
                        src_r = new_r + 4
                        # Copy the entire cross pattern
                        for c in range(cols):
                            if grid[src_r][c] == cross_color or result[src_r][c] == cross_color:
                                result[new_r][c] = cross_color

                    cross_rows.extend(new_cross_rows)
                    if not new_cross_rows:
                        break

    # Clear last row
    result[-1] = [background] * cols

    return result
