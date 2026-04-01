def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid mostly filled with 8s (background) with some non-8 colored components
    2. Output creates L-shapes at the four corners of the grid
    3. A separator row (at N//2) and column (at N//2) filled with 8s divides the grid
    4. Each color maps to a corner based on which quadrant has the most cells of that color
    5. L-shapes consist of top/bottom edge and left/right edge portions

    Procedure:
    1. Divide grid into four quadrants (excluding separator lines)
    2. For each quadrant, count how many cells of each color appear in it
    3. Assign each color to the quadrant where it has the most presence
    4. Handle assignment conflicts by proximity or color priority
    5. Draw L-shapes in the assigned corners
    """

    rows, cols = len(grid), len(grid[0])
    result = [[8 for _ in range(cols)] for _ in range(rows)]
    center_r, center_c = rows // 2, cols // 2

    # Find all colors and count their presence in each quadrant
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                color = grid[r][c]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((r, c))

    # For each color, count presence in each quadrant (excluding separator)
    color_quadrant_counts = {}
    for color, positions in color_positions.items():
        counts = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
        for r, c in positions:
            if r == center_r or c == center_c:
                continue  # Skip separator
            if r < center_r and c < center_c:
                counts["top_left"] += 1
            elif r < center_r and c > center_c:
                counts["top_right"] += 1
            elif r > center_r and c < center_c:
                counts["bottom_left"] += 1
            elif r > center_r and c > center_c:
                counts["bottom_right"] += 1
        color_quadrant_counts[color] = counts

    # Assign each quadrant to the color with most cells in it
    quadrant_assignments = {}
    for quadrant in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        best_color = None
        max_count = 0
        for color, counts in color_quadrant_counts.items():
            if counts[quadrant] > max_count:
                max_count = counts[quadrant]
                best_color = color
        if best_color:
            quadrant_assignments[quadrant] = best_color

    # Draw L-shapes
    if "top_left" in quadrant_assignments:
        color = quadrant_assignments["top_left"]
        for c in range(center_c):
            result[0][c] = color
        for r in range(center_r):
            result[r][0] = color

    if "top_right" in quadrant_assignments:
        color = quadrant_assignments["top_right"]
        for c in range(center_c + 1, cols):
            result[0][c] = color
        for r in range(center_r):
            result[r][cols - 1] = color

    if "bottom_left" in quadrant_assignments:
        color = quadrant_assignments["bottom_left"]
        for c in range(center_c):
            result[rows - 1][c] = color
        for r in range(center_r + 1, rows):
            result[r][0] = color

    if "bottom_right" in quadrant_assignments:
        color = quadrant_assignments["bottom_right"]
        for c in range(center_c + 1, cols):
            result[rows - 1][c] = color
        for r in range(center_r + 1, rows):
            result[r][cols - 1] = color

    # Handle right edge extension when top_right exists but bottom_right doesn't
    if "top_right" in quadrant_assignments:
        color = quadrant_assignments["top_right"]
        for r in range(center_r, rows):
            if r != center_r:
                result[r][cols - 1] = color
        # Also extend bottom row if no bottom_right
        if "bottom_right" not in quadrant_assignments:
            for c in range(center_c + 1, cols):
                result[rows - 1][c] = color

    return result
