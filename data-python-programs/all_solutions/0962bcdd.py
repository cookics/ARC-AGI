def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing cross-shaped patterns with a center cell and 4 orthogonal neighbors.
    2. Each cross pattern has all non-zero values where the center cell differs from the 4 surrounding cells.
    3. All 4 surrounding cells in a cross have the same color value.
    4. The output transforms each cross into a 5x5 expansion using a specific template pattern.
    5. The template uses center color (C) and surrounding color (S) in this diamond pattern: C-0-S-0-C / 0-C-S-C-0 / S-S-C-S-S / 0-C-S-C-0 / C-0-S-0-C.

    Procedure:
    1. Find all cross patterns in the input grid by checking each cell and its 4 orthogonal neighbors.
    2. For each valid cross pattern, extract the center color and surrounding color values.
    3. Apply the 5x5 expansion template centered on each cross position.
    4. Handle overlapping regions by overwriting previous values in the result grid.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all cross patterns
    crosses = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = grid[r][c]
            up = grid[r - 1][c]
            down = grid[r + 1][c]
            left = grid[r][c - 1]
            right = grid[r][c + 1]

            # Check if this forms a cross pattern (center + 4 neighbors all non-zero)
            if (
                center != 0
                and up != 0
                and down != 0
                and left != 0
                and right != 0
                and up == down == left == right
                and up != center
            ):
                crosses.append((r, c, center, up))

    # Apply 5x5 expansion for each cross
    # Template: C=center_color, S=surrounding_color, 0=background
    # C 0 S 0 C
    # 0 C S C 0
    # S S C S S
    # 0 C S C 0
    # C 0 S 0 C

    expansion_template = [
        [1, 0, 2, 0, 1],  # 1=center_color, 0=background(0), 2=surrounding_color
        [0, 1, 2, 1, 0],
        [2, 2, 1, 2, 2],
        [0, 1, 2, 1, 0],
        [1, 0, 2, 0, 1],
    ]

    for cross_r, cross_c, center_color, surrounding_color in crosses:
        # Apply 5x5 pattern centered on the cross
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                new_r, new_c = cross_r + dr, cross_c + dc
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    template_val = expansion_template[dr + 2][dc + 2]
                    if template_val == 1:
                        result[new_r][new_c] = center_color
                    elif template_val == 0:
                        result[new_r][new_c] = 0  # background
                    elif template_val == 2:
                        result[new_r][new_c] = surrounding_color

    return result
