def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is 15x15 grid with horizontal line of 5s dividing it into top and bottom sections
    2. Top section has patterns with colors 1, 2, and a third color
    3. Bottom section has pattern with color 1
    4. Output combines the bottom section's 1s pattern and top section's third color pattern
    5. Order depends on bottom pattern's center column: if <= 6, top first; if > 6, bottom first

    Procedure:
    1. Find the divider line (row of 5s)
    2. Extract patterns from top and bottom sections
    3. Identify the third color in top section (not 1 or 2)
    4. Determine order based on bottom pattern center column
    5. Position both patterns in the output grid using leftmost pattern's column position
    """

    # Find divider line (row of 5s)
    divider_row = -1
    for i in range(len(grid)):
        if all(cell == 5 for cell in grid[i]):
            divider_row = i
            break

    # Extract patterns from top section (above divider)
    top_section = grid[:divider_row]

    # Extract patterns from bottom section (below divider)
    bottom_section = grid[divider_row + 1 :]

    # Find all colors in top section
    top_colors = set()
    for row in top_section:
        for cell in row:
            if cell != 0:
                top_colors.add(cell)

    # Find the third color (not 1 or 2)
    third_color = None
    for color in top_colors:
        if color not in [1, 2]:
            third_color = color
            break

    # Extract bottom section 1s pattern
    bottom_pattern = []
    for r in range(len(bottom_section)):
        for c in range(len(bottom_section[0])):
            if bottom_section[r][c] == 1:
                bottom_pattern.append((r, c))

    # Extract top section third color pattern
    top_pattern = []
    for r in range(len(top_section)):
        for c in range(len(top_section[0])):
            if top_section[r][c] == third_color:
                top_pattern.append((r, c))

    # Find bounding boxes
    if bottom_pattern:
        bottom_min_r = min(r for r, c in bottom_pattern)
        bottom_max_r = max(r for r, c in bottom_pattern)
        bottom_min_c = min(c for r, c in bottom_pattern)
        bottom_max_c = max(c for r, c in bottom_pattern)
        bottom_center_c = (bottom_min_c + bottom_max_c) // 2

    if top_pattern:
        top_min_r = min(r for r, c in top_pattern)
        top_max_r = max(r for r, c in top_pattern)
        top_min_c = min(c for r, c in top_pattern)
        top_max_c = max(c for r, c in top_pattern)

    # Determine order based on bottom pattern center column
    top_first = bottom_center_c <= 6

    # Determine reference column (leftmost pattern)
    ref_col = min(bottom_min_c, top_min_c)

    # Create output grid (9 rows x 15 cols)
    result = [[0] * 15 for _ in range(9)]

    # Bottom pattern keeps its original row position
    bottom_height = bottom_max_r - bottom_min_r + 1

    if top_first:
        # Position top pattern first, then bottom at its original position
        top_start_row = bottom_min_r - (top_max_r - top_min_r + 1)
        for r, c in top_pattern:
            adjusted_c = c - top_min_c + ref_col
            result[r - top_min_r + top_start_row][adjusted_c] = third_color

        # Position bottom pattern at its original position
        for r, c in bottom_pattern:
            adjusted_c = c - bottom_min_c + ref_col
            result[r][adjusted_c] = 1
    else:
        # Position bottom pattern at its original position
        for r, c in bottom_pattern:
            adjusted_c = c - bottom_min_c + ref_col
            result[r][adjusted_c] = 1

        # Position top pattern after bottom
        top_start_row = bottom_max_r + 1
        for r, c in top_pattern:
            adjusted_c = c - top_min_c + ref_col
            result[r - top_min_r + top_start_row][adjusted_c] = third_color

    return result
