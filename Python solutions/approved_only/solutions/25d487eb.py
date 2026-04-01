def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing mostly zeros with a structure made of two colors.
    2. Output is the same grid but with the minority color extended in a line.
    3. The main structure consists of one dominant color with one special (minority) color cell.
    4. The special color cell extends away from the main structure toward empty space.
    5. Extension goes to the grid boundary in the direction with most available space.

    Procedure:
    1. Find the special color cell (minority color based on frequency count).
    2. Determine the bounds of the main structure to understand spatial relationships.
    3. Calculate available space in each cardinal direction from the special cell.
    4. Choose the direction with maximum available space for extension.
    5. Extend the special color in a straight line in the chosen direction to the boundary.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy grid

    # Find all non-zero cells and count colors
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                color_counts[color] = color_counts.get(color, 0) + 1

    if len(color_counts) < 2:
        return result

    # Identify main color (most frequent) and special color (least frequent)
    main_color = max(color_counts.keys(), key=lambda x: color_counts[x])
    special_color = min(color_counts.keys(), key=lambda x: color_counts[x])

    # Find position of special color
    special_r, special_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == special_color:
                special_r, special_c = r, c
                break
        if special_r is not None:
            break

    if special_r is None:
        return result

    # Find bounds of main structure
    main_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == main_color:
                main_positions.append((r, c))

    min_r = min(r for r, c in main_positions)
    max_r = max(r for r, c in main_positions)
    min_c = min(c for r, c in main_positions)
    max_c = max(c for r, c in main_positions)

    # Count available space to boundary in each direction from special cell
    directions = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]

    best_dir = None
    max_space = 0

    for dir_name, dr, dc in directions:
        if dir_name == "up":
            # Count space from special cell to top boundary
            space_count = special_r
        elif dir_name == "down":
            # Count space from special cell to bottom boundary
            space_count = rows - 1 - special_r
        elif dir_name == "left":
            # Count space from special cell to left boundary
            space_count = special_c
        elif dir_name == "right":
            # Count space from special cell to right boundary
            # For rightward extension, if inside main structure, start beyond it
            if special_c <= max_c:
                space_count = cols - 1 - max_c
            else:
                space_count = cols - 1 - special_c

        if space_count > max_space:
            max_space = space_count
            best_dir = (dir_name, dr, dc)

    # Extend in the best direction
    if best_dir is not None:
        dir_name, dr, dc = best_dir

        if dir_name == "up":
            for r in range(0, special_r):
                if result[r][special_c] == 0:  # Only extend into empty spaces
                    result[r][special_c] = special_color
        elif dir_name == "down":
            for r in range(special_r + 1, rows):
                if result[r][special_c] == 0:  # Only extend into empty spaces
                    result[r][special_c] = special_color
        elif dir_name == "left":
            for c in range(0, special_c):
                if result[special_r][c] == 0:  # Only extend into empty spaces
                    result[special_r][c] = special_color
        elif dir_name == "right":
            # For rightward, start from beyond main structure if applicable
            start_c = (
                max(max_c + 1, special_c + 1) if special_c <= max_c else special_c + 1
            )
            for c in range(start_c, cols):
                if result[special_r][c] == 0:  # Only extend into empty spaces
                    result[special_r][c] = special_color

    return result
