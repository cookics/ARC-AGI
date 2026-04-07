def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input grid contains a horizontal line that spans the entire width with the same non-zero color.
    2. The input grid contains a vertical line that spans the entire height with the same color as the horizontal line.
    3. These perpendicular dividing lines create exactly 4 quadrants in the grid.
    4. Each quadrant contains colored patterns (non-zero values different from the dividing line color).
    5. The output is always a 6x6 grid formed by combining 3x3 regions from each quadrant.
    6. The 3x3 regions are arranged in a 2x2 layout: top-left and top-right quadrants form the top half, bottom-left and bottom-right quadrants form the bottom half.

    Procedure:
    1. Find the horizontal dividing line by locating a row where all cells have the same non-zero color.
    2. Find the vertical dividing line by locating a column where all cells have the same color as the horizontal line.
    3. Define the 4 quadrants using the intersection of the horizontal and vertical dividing lines.
    4. For each quadrant, identify all non-zero pixels that are not part of the dividing lines.
    5. Calculate the bounding box of these pixels and extract a 3x3 region starting from the minimum coordinates.
    6. Combine the four 3x3 regions by placing top-left and top-right side by side for rows 0-2, and bottom-left and bottom-right side by side for rows 3-5.
    """

    rows, cols = len(grid), len(grid[0])

    # Find horizontal dividing line (row that is entirely the same non-zero color)
    horizontal_line_row = -1
    dividing_color = -1
    for r in range(rows):
        color = grid[r][0]
        if color != 0 and all(grid[r][c] == color for c in range(cols)):
            horizontal_line_row = r
            dividing_color = color
            break

    # Find vertical dividing line (column that is entirely the same color as horizontal line)
    vertical_line_col = -1
    for c in range(cols):
        if all(grid[r][c] == dividing_color for r in range(rows)):
            vertical_line_col = c
            break

    # Define quadrants
    quadrants = [
        (0, horizontal_line_row, 0, vertical_line_col),  # top-left
        (0, horizontal_line_row, vertical_line_col + 1, cols),  # top-right
        (horizontal_line_row + 1, rows, 0, vertical_line_col),  # bottom-left
        (horizontal_line_row + 1, rows, vertical_line_col + 1, cols),  # bottom-right
    ]

    def extract_3x3_from_quadrant(r1, r2, c1, c2):
        # Find all non-zero, non-dividing-line pixels in this quadrant
        pixels = []
        for r in range(r1, r2):
            for c in range(c1, c2):
                if grid[r][c] != 0 and grid[r][c] != dividing_color:
                    pixels.append((r, c))

        if not pixels:
            # Return empty 3x3 if no pixels found
            return [[0] * 3 for _ in range(3)]

        # Find bounding box
        min_r = min(r for r, c in pixels)
        max_r = max(r for r, c in pixels)
        min_c = min(c for r, c in pixels)
        max_c = max(c for r, c in pixels)

        # Extract 3x3 region
        result = []
        for r in range(min_r, min_r + 3):
            row = []
            for c in range(min_c, min_c + 3):
                if r < rows and c < cols:
                    row.append(grid[r][c] if grid[r][c] != dividing_color else 0)
                else:
                    row.append(0)
            result.append(row)

        return result

    # Extract 3x3 from each quadrant
    quadrant_3x3s = []
    for r1, r2, c1, c2 in quadrants:
        quadrant_3x3s.append(extract_3x3_from_quadrant(r1, r2, c1, c2))

    # Arrange in 2x2 grid to form 6x6 output
    output = []

    # Top half (rows 0-2): top-left and top-right side by side
    for r in range(3):
        row = quadrant_3x3s[0][r] + quadrant_3x3s[1][r]
        output.append(row)

    # Bottom half (rows 3-5): bottom-left and bottom-right side by side
    for r in range(3):
        row = quadrant_3x3s[2][r] + quadrant_3x3s[3][r]
        output.append(row)

    return output
