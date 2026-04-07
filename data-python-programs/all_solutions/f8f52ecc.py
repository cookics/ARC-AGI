def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The task creates a vertical spine at the leftmost column and horizontal
    branches for each row. The spine only covers rows that need connection.

    Procedure:
    1. Find spine column (leftmost column containing the color)
    2. Determine spine start row (skip already-connected rows)
    3. Create vertical spine and horizontal branches
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all non-background, non-obstacle colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in [1, 8]:
                colors.add(grid[r][c])

    for color in colors:
        positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    positions.append((r, c))

        if len(positions) <= 1:
            continue

        # Find spine column using hybrid approach
        col_counts = {}
        for r, c in positions:
            col_counts[c] = col_counts.get(c, 0) + 1

        max_count = max(col_counts.values())
        # If one column has significantly more instances (more than half), use it
        if max_count > len(positions) // 2:
            spine_col = max(col_counts.keys(), key=lambda c: col_counts[c])
        else:
            # Otherwise use leftmost column
            spine_col = min(col_counts.keys())

        # Find rows containing this color
        rows_with_color = sorted(set(pos[0] for pos in positions))

        # Determine spine start (skip first row if it's already connected to second row)
        spine_start = rows_with_color[0]
        spine_end = rows_with_color[-1]

        if len(rows_with_color) >= 2:
            r1, r2 = rows_with_color[0], rows_with_color[1]
            if abs(r1 - r2) == 1:  # Adjacent rows
                cols_r1 = [pos[1] for pos in positions if pos[0] == r1]
                cols_r2 = [pos[1] for pos in positions if pos[0] == r2]
                # If they share a column, they're already connected
                if any(c in cols_r2 for c in cols_r1):
                    spine_start = r2

        # Create vertical spine
        if spine_start <= spine_end:
            for r in range(spine_start, spine_end + 1):
                if result[r][spine_col] == 1:
                    result[r][spine_col] = color

        # Create horizontal branches only for rows that need them
        for r in rows_with_color:
            cols_in_row = [pos[1] for pos in positions if pos[0] == r]

            # Only create horizontal branches if:
            # 1. The row has multiple instances of the color, OR
            # 2. The row has instances not in the spine column and needs connection to spine
            needs_horizontal_branch = len(cols_in_row) > 1 or (
                len(cols_in_row) == 1
                and spine_col not in cols_in_row
                and spine_start <= r <= spine_end
            )

            if needs_horizontal_branch:
                min_col = min(cols_in_row)
                max_col = max(cols_in_row)

                # Connect all positions in this row horizontally
                for c in range(min_col, max_col + 1):
                    if result[r][c] == 1:
                        result[r][c] = color

                # Connect from spine to this row - handle both directions
                if spine_col < min_col:
                    for c in range(spine_col, min_col + 1):
                        if result[r][c] == 1:
                            result[r][c] = color
                elif spine_col > max_col:
                    for c in range(max_col, spine_col + 1):
                        if result[r][c] == 1:
                            result[r][c] = color

    return result
