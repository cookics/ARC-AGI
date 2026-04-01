def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid containing exactly two colored patterns (non-zero values)
    2. Output is a 3x3 grid combining these two patterns
    3. Patterns are extracted relative to their bounding boxes
    4. Placement depends on:
       - If one pattern is exactly 3x3, it fills the output, other pattern overlays
       - Otherwise, patterns are placed based on their spatial separation:
         * Horizontally separated → side-by-side (left stays left, right stays right)
         * Vertically separated → top-bottom (inverted: bottom→top, top→bottom)

    Procedure:
    1. Extract both colored patterns and compute their bounding boxes
    2. Calculate centroid positions to determine relative placement
    3. Place patterns in 3x3 output based on separation direction
    4. Overlay patterns (non-zero values from both contribute to output)
    """

    # Find all non-zero colors
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)

    colors = list(colors)

    # Extract patterns for each color
    patterns = {}
    for color in colors:
        cells = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell == color:
                    cells.append((i, j))

        min_row = min(r for r, c in cells)
        max_row = max(r for r, c in cells)
        min_col = min(c for r, c in cells)
        max_col = max(c for r, c in cells)

        height = max_row - min_row + 1
        width = max_col - min_col + 1

        # Extract pattern relative to bounding box
        pattern = [[0] * width for _ in range(height)]
        for r, c in cells:
            pattern[r - min_row][c - min_col] = color

        avg_row = sum(r for r, c in cells) / len(cells)
        avg_col = sum(c for r, c in cells) / len(cells)

        patterns[color] = {
            'pattern': pattern,
            'height': height,
            'width': width,
            'avg_row': avg_row,
            'avg_col': avg_col,
        }

    # Create 3x3 result
    result = [[0] * 3 for _ in range(3)]

    # Check if one pattern is exactly 3x3
    full_pattern = None
    other_pattern = None
    for color in colors:
        if patterns[color]['height'] == 3 and patterns[color]['width'] == 3:
            full_pattern = color
        else:
            other_pattern = color

    if full_pattern is not None:
        # Place the 3x3 pattern first
        p = patterns[full_pattern]
        for i in range(3):
            for j in range(3):
                result[i][j] = p['pattern'][i][j]

        # Place the other pattern based on its position
        if other_pattern is not None:
            p = patterns[other_pattern]
            # Smaller avg_row → bottom alignment (inverted)
            if p['avg_row'] < 5:
                row_offset = 3 - p['height']
            else:
                row_offset = 0

            # Left alignment
            col_offset = 0

            for i in range(p['height']):
                for j in range(p['width']):
                    if p['pattern'][i][j] != 0:
                        result[row_offset + i][col_offset + j] = p['pattern'][i][j]
    else:
        # Neither pattern is 3x3, use relative placement
        col_diff = abs(patterns[colors[0]]['avg_col'] - patterns[colors[1]]['avg_col'])
        row_diff = abs(patterns[colors[0]]['avg_row'] - patterns[colors[1]]['avg_row'])

        if col_diff > row_diff:
            # Horizontal placement (side-by-side)
            if patterns[colors[0]]['avg_col'] < patterns[colors[1]]['avg_col']:
                left_color, right_color = colors[0], colors[1]
            else:
                left_color, right_color = colors[1], colors[0]

            # Place left pattern at left
            p = patterns[left_color]
            for i in range(p['height']):
                for j in range(p['width']):
                    if p['pattern'][i][j] != 0:
                        result[i][j] = p['pattern'][i][j]

            # Place right pattern at right
            p = patterns[right_color]
            col_offset = 3 - p['width']
            for i in range(p['height']):
                for j in range(p['width']):
                    if p['pattern'][i][j] != 0:
                        result[i][col_offset + j] = p['pattern'][i][j]
        else:
            # Vertical placement (top-bottom, inverted)
            if patterns[colors[0]]['avg_row'] > patterns[colors[1]]['avg_row']:
                top_color, bottom_color = colors[0], colors[1]
            else:
                top_color, bottom_color = colors[1], colors[0]

            # Place top pattern at top
            p = patterns[top_color]
            for i in range(p['height']):
                for j in range(p['width']):
                    if p['pattern'][i][j] != 0:
                        result[i][j] = p['pattern'][i][j]

            # Place bottom pattern at bottom
            p = patterns[bottom_color]
            row_offset = 3 - p['height']
            for i in range(p['height']):
                for j in range(p['width']):
                    if p['pattern'][i][j] != 0:
                        result[row_offset + i][j] = p['pattern'][i][j]

    return result
