def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Pattern has outer frame color and inner marker color
    2. A border is added around the pattern using inversion for straight edges
    3. For diagonal positions, use diagonal-to-corner marking
    4. This handles both cross/rectangular and diamond shapes

    Procedure:
    1. Find bounding box and identify colors
    2. Add inverted border rows/columns (handles straight edges)
    3. Add diagonal marks at corners (handles diamond shapes)
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all non-background positions
    non_bg_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 7]
    if not non_bg_positions:
        return result

    # Find bounding box
    min_r = min(p[0] for p in non_bg_positions)
    max_r = max(p[0] for p in non_bg_positions)
    min_c = min(p[1] for p in non_bg_positions)
    max_c = max(p[1] for p in non_bg_positions)

    # Identify colors
    color_counts = {}
    for r, c in non_bg_positions:
        color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1

    inner_color = min(color_counts.keys(), key=lambda x: color_counts[x])
    outer_color = [c for c in color_counts if c != inner_color][0]

    # Detect shape type: check if frame colors are at bbox corners
    corners_with_frame = 0
    for r, c in [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]:
        if grid[r][c] == outer_color:
            corners_with_frame += 1

    is_diamond = (corners_with_frame == 0)

    if is_diamond:
        # Diamond shape: mark diagonals that move away from bbox center
        center_r = (min_r + max_r) / 2
        center_c = (min_c + max_c) / 2

        outer_positions = [(r, c) for r, c in non_bg_positions if grid[r][c] == outer_color]
        for r, c in outer_positions:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 7:
                    # Check if diagonal moves away from center in BOTH dimensions
                    moves_away_r = (r < center_r and nr < r) or (r > center_r and nr > r)
                    moves_away_c = (c < center_c and nc < c) or (c > center_c and nc > c)
                    moves_away = moves_away_r and moves_away_c

                    at_corner = ((nr == min_r or nr == max_r) and (nc == min_c or nc == max_c))
                    outside = (nr < min_r or nr > max_r or nc < min_c or nc > max_c)

                    if (at_corner or outside) and moves_away:
                        result[nr][nc] = inner_color
    else:
        # Cross/rectangular shape: use border inversion
        def invert(val):
            return inner_color if val == 7 else 7

        # Add border row above
        if min_r > 0:
            for c in range(min_c, max_c + 1):
                result[min_r - 1][c] = invert(grid[min_r][c])

        # Add border row below
        if max_r < rows - 1:
            for c in range(min_c, max_c + 1):
                result[max_r + 1][c] = invert(grid[max_r][c])

        # Add border column left
        if min_c > 0:
            for r in range(min_r, max_r + 1):
                result[r][min_c - 1] = invert(grid[r][min_c])

        # Add border column right
        if max_c < cols - 1:
            for r in range(min_r, max_r + 1):
                result[r][max_c + 1] = invert(grid[r][max_c])

        # Handle corners with diagonal rule
        if min_r > 0 and min_c > 0:
            result[min_r - 1][min_c - 1] = 7 if grid[min_r][min_c] == 7 else inner_color
        if min_r > 0 and max_c < cols - 1:
            result[min_r - 1][max_c + 1] = 7 if grid[min_r][max_c] == 7 else inner_color
        if max_r < rows - 1 and min_c > 0:
            result[max_r + 1][min_c - 1] = 7 if grid[max_r][min_c] == 7 else inner_color
        if max_r < rows - 1 and max_c < cols - 1:
            result[max_r + 1][max_c + 1] = 7 if grid[max_r][max_c] == 7 else inner_color

    return result
