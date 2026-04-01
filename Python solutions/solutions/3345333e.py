def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a solid rectangular block of one color (e.g., 3s or 1s)
    2. Input contains a non-solid scattered pattern of another color (e.g., 2s or 6s)
    3. Output shows the scattered pattern reflected vertically to create symmetry
    4. The solid block is replaced/removed by this reflection process

    Procedure:
    1. Identify the two non-zero colors (solid block vs pattern)
    2. Determine which is solid (forms perfect rectangle) vs scattered pattern
    3. Find the column range (bounding box) of the pattern color
    4. Calculate the vertical axis of symmetry (center of column range)
    5. Clear the solid block and reflect the pattern across the axis
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all non-zero colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])

    if len(colors) != 2:
        return result

    color1, color2 = sorted(colors)

    # Check which color forms a solid rectangular block
    def is_solid_block(color):
        positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == color]
        if not positions:
            return False

        min_r = min(r for r, c in positions)
        max_r = max(r for r, c in positions)
        min_c = min(c for r, c in positions)
        max_c = max(c for r, c in positions)

        # Check if all cells in this rectangle have the color
        expected_count = (max_r - min_r + 1) * (max_c - min_c + 1)
        return len(positions) == expected_count

    is_solid1 = is_solid_block(color1)
    is_solid2 = is_solid_block(color2)

    if is_solid1 and not is_solid2:
        solid_color = color1
        pattern_color = color2
    elif is_solid2 and not is_solid1:
        solid_color = color2
        pattern_color = color1
    else:
        # Use count heuristic if both or neither are solid
        count1 = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == color1)
        count2 = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == color2)
        if count1 < count2:
            solid_color = color1
            pattern_color = color2
        else:
            solid_color = color2
            pattern_color = color1

    # Find bounding box of pattern color
    pattern_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == pattern_color]
    if not pattern_positions:
        return result

    min_col = min(c for r, c in pattern_positions)
    max_col = max(c for r, c in pattern_positions)

    # Calculate vertical axis of symmetry (center of column range)
    axis = (min_col + max_col) / 2.0

    # Clear solid block in result
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == solid_color:
                result[r][c] = 0

    # Reflect pattern across the axis
    for r, c in pattern_positions:
        reflected_c = int(round(2 * axis - c))
        if 0 <= reflected_c < cols:
            result[r][reflected_c] = pattern_color

    return result
