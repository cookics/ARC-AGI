def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with obstacles/frames (non-zero colored cells)
    2. Output fills cells with color 4 based on:
       - Interior regions of frames (cells surrounded by obstacles)
       - Diagonal lines from corners
       - Horizontal/vertical fills triggered by diagonal-obstacle interactions
       - Converging pyramid patterns
       - Spreading patterns through openings

    Procedure:
    1. Fill interior regions (cells within frame structures)
    2. Trace diagonals from corners
    3. Analyze obstacle structure for pattern detection
    4. Apply fills based on interactions
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find obstacles
    obstacles = set()
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0:
                obstacles.add((i, j))

    if not obstacles:
        return result

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def is_empty(r, c):
        return is_valid(r, c) and (r, c) not in obstacles and result[r][c] == 0

    # Get obstacle bounds
    min_r = min(r for r, c in obstacles)
    max_r = max(r for r, c in obstacles)
    min_c = min(c for r, c in obstacles)
    max_c = max(c for r, c in obstacles)

    # Fill interior regions - cells surrounded by obstacles in all 4 directions
    # OR cells that are in the interior of the bounding box
    for i in range(min_r, max_r + 1):
        for j in range(min_c, max_c + 1):
            if (i, j) not in obstacles:
                # Check if surrounded by obstacles in all 4 directions
                has_obstacle_north = any((r, j) in obstacles for r in range(min_r, i))
                has_obstacle_south = any(
                    (r, j) in obstacles for r in range(i + 1, max_r + 1)
                )
                has_obstacle_west = any((i, c) in obstacles for c in range(min_c, j))
                has_obstacle_east = any(
                    (i, c) in obstacles for c in range(j + 1, max_c + 1)
                )

                # Count how many directions have obstacles
                directions_blocked = sum(
                    [
                        has_obstacle_north,
                        has_obstacle_south,
                        has_obstacle_west,
                        has_obstacle_east,
                    ]
                )

                # Fill if surrounded on all 4 sides, or on 3 sides and in interior area
                if directions_blocked == 4:
                    result[i][j] = 4
                elif directions_blocked >= 3:
                    # Allow if not on the outer edge or if clearly interior
                    on_outer_edge = (i == min_r and (j == min_c or j == max_c)) or (
                        i == max_r and (j == min_c or j == max_c)
                    )
                    if not on_outer_edge:
                        result[i][j] = 4

    # Analyze row structure
    row_structure = {}
    for r in range(rows):
        row_obs = [c for c in range(cols) if (r, c) in obstacles]
        if row_obs:
            row_structure[r] = {
                "min": min(row_obs),
                "max": max(row_obs),
                "count": len(row_obs),
            }

    # Determine pattern type
    center_c = (min_c + max_c) / 2
    center_r = (min_r + max_r) / 2

    # Check if obstacles are in bottom portion and centered (converging pyramid pattern)
    is_bottom_centered = (min_r >= rows * 0.5) and (
        abs(center_c - cols / 2) < cols * 0.4
    )

    if is_bottom_centered:
        # PATTERN: Converging diagonal lines (not filled pyramid)
        center_col = round(center_c)

        # Fill center vertical line
        for r in range(min_r):
            if is_empty(r, center_col):
                result[r][center_col] = 4

        # Draw diagonal lines converging from corners toward center
        # Start from row 1 (not row 0) to create the proper pattern
        # From left corner diagonal
        r, c = 1, 0
        while r < min_r and c < center_col and is_empty(r, c):
            result[r][c] = 4
            r += 1
            c += 1

        # From right corner diagonal
        r, c = 1, cols - 1
        while r < min_r and c > center_col and is_empty(r, c):
            result[r][c] = 4
            r += 1
            c -= 1

    else:
        # PATTERN: Diagonal-triggered fills

        # Diagonal from top-left
        r, c = 0, 0
        diagonal_end = -1
        while is_empty(r, c):
            result[r][c] = 4
            diagonal_end = r
            r += 1
            c += 1

        # Check for rows with one-sided walls - fill horizontally
        if diagonal_end >= 0:
            for r in range(diagonal_end, rows):
                if r in row_structure:
                    min_obs = row_structure[r]["min"]
                    max_obs = row_structure[r]["max"]
                    # If row has obstacle on right edge but NO obstacle on left side
                    # Check if there are no obstacles in the left half of the row
                    left_half_clear = min_obs > cols // 2
                    if max_obs >= cols - 1 and left_half_clear:
                        for c in range(max_obs):
                            if is_empty(r, c):
                                result[r][c] = 4

        # Diagonal from bottom going up-right
        # Start position aligns with the obstacle pattern
        r, c = rows - 1, min_r
        while r >= 0 and c < cols and is_empty(r, c):
            result[r][c] = 4
            r -= 1
            c += 1

        # Note: Top-right diagonal removed as it's not needed for this pattern

        # Check for spreading through openings below obstacle
        if max_r < rows - 1:
            # Find cells at bottom edge of obstacle region that are filled
            opening_cols = []
            for c in range(min_c, max_c + 1):
                if (max_r, c) not in obstacles and result[max_r][c] == 4:
                    opening_cols.append(c)

            if opening_cols:
                # Spread downward and outward
                current_row_cols = set(opening_cols)

                # Extend one level around opening
                extended = set()
                for c in opening_cols:
                    for dc in range(-1, 2):
                        nc = c + dc
                        if min_c <= nc <= max_c:
                            extended.add(nc)
                current_row_cols = extended

                # Fill subsequent rows with spreading pattern
                for r in range(max_r + 1, rows):
                    dist = r - max_r

                    # Fill current level
                    for c in current_row_cols:
                        if is_empty(r, c):
                            result[r][c] = 4

                    # Spread for next row - outer cells spread diagonally, inner go straight
                    if r < rows - 1:
                        next_cols = set()
                        sorted_cols = sorted(current_row_cols)
                        if sorted_cols:
                            min_col = sorted_cols[0]
                            max_col = sorted_cols[-1]
                            for c in sorted_cols:
                                # Outer cells spread diagonally
                                if c == min_col and c > 0:
                                    next_cols.add(c - 1)  # Spread left
                                elif c == max_col and c < cols - 1:
                                    next_cols.add(c + 1)  # Spread right
                                else:
                                    # Inner cells go straight down
                                    next_cols.add(c)
                        current_row_cols = next_cols

    return result
