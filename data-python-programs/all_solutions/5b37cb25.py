def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has borders, background, and payload blobs
    2. Payload blobs have concave indentations (notches)
    3. Specific indentation patterns get filled with cross patterns
    4. Cross color = nearest border

    Procedure:
    1. Identify background color
    2. For each background cell, check if it's a notch center
    3. Notch = background with payload on 3 sides forming T or L shape
    4. Fill notch with cross using nearest border color
    """
    rows = len(grid)
    cols = len(grid[0])

    # Extract border colors
    top_color = grid[0][1]
    bottom_color = grid[-1][1]
    left_color = grid[1][0]
    right_color = grid[1][-1]

    # Identify background
    color_count = {}
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            color = grid[r][c]
            color_count[color] = color_count.get(color, 0) + 1

    background_color = max(color_count, key=color_count.get)

    result = [row[:] for row in grid]

    # Find notches: background cells with exactly 3 payload neighbors in T configuration
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if grid[r][c] != background_color:
                continue

            # Check 4-neighbors
            up = grid[r-1][c] != background_color
            down = grid[r+1][c] != background_color
            left = grid[r][c-1] != background_color
            right = grid[r][c+1] != background_color

            # Count payload neighbors
            count = sum([up, down, left, right])

            # Check for T-shape (3 payload neighbors)
            if count == 3:
                # Determine which border is closest
                dist_top = r
                dist_bottom = (rows - 1) - r
                dist_left = c
                dist_right = (cols - 1) - c

                min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

                if dist_top == min_dist:
                    fill_color = top_color
                elif dist_bottom == min_dist:
                    fill_color = bottom_color
                elif dist_left == min_dist:
                    fill_color = left_color
                else:
                    fill_color = right_color

                # Fill cross: center + all 4 adjacent cells if background
                result[r][c] = fill_color

                # Fill all 4 directions (1 cell each)
                if r > 0 and grid[r-1][c] == background_color:
                    result[r-1][c] = fill_color
                if r < rows-1 and grid[r+1][c] == background_color:
                    result[r+1][c] = fill_color
                if c > 0 and grid[r][c-1] == background_color:
                    result[r][c-1] = fill_color
                if c < cols-1 and grid[r][c+1] == background_color:
                    result[r][c+1] = fill_color

    return result
