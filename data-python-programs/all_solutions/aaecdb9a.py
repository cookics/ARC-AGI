def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid where 7 is the background color
    2. Output is always 5 columns wide, with variable number of rows
    3. Each non-7 color has a cross/plus pattern (cell with 4 same-color orthogonal neighbors)
    4. The number of cross patterns a color has determines how many rows it occupies vertically
    5. All colors appear in the bottom row at positions determined by their spatial location
    6. Colors with crosses extend upward from the bottom row

    Procedure:
    1. Find all non-7 colors and their positions
    2. Count cross patterns for each color
    3. Calculate spatial properties (average column position) for each color
    4. Assign each color to one of 5 output columns based on spatial ordering
    5. Determine output height based on maximum cross count
    6. Place colors vertically in their assigned columns based on cross count
    7. Fill the bottom row with all colors at their assigned positions
    """

    rows, cols = len(grid), len(grid[0])

    # Find all non-7 colors and their positions
    color_info = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7:
                color = grid[r][c]
                if color not in color_info:
                    color_info[color] = {'positions': [], 'crosses': 0}
                color_info[color]['positions'].append((r, c))

    if not color_info:
        return [[7]]

    # Count cross patterns for each color
    def has_cross_at(r, c, color):
        """Check if color forms a cross pattern centered at (r, c)"""
        if r < 1 or r >= rows - 1 or c < 1 or c >= cols - 1:
            return False
        return (grid[r][c] == color and
                grid[r-1][c] == color and
                grid[r+1][c] == color and
                grid[r][c-1] == color and
                grid[r][c+1] == color)

    for color in color_info:
        crosses = set()
        for r, c in color_info[color]['positions']:
            if has_cross_at(r, c, color):
                crosses.add((r, c))
        color_info[color]['crosses'] = len(crosses)

        # Calculate average column position
        cols_list = [c for r, c in color_info[color]['positions']]
        color_info[color]['avg_col'] = sum(cols_list) / len(cols_list)

    # Sort colors by average column position to assign to output columns
    colors_by_position = sorted(color_info.keys(), key=lambda c: color_info[c]['avg_col'])

    # Determine output height: max crosses among all colors + 1 (for bottom row)
    max_crosses = max(color_info[c]['crosses'] for c in color_info)
    output_height = max_crosses + 1

    # Create output grid
    result = [[7] * 5 for _ in range(output_height)]

    # Assign each color to a column (0-4) based on sorted order
    # We have up to 5 colors and 5 columns
    color_to_column = {}
    for i, color in enumerate(colors_by_position):
        if i < 5:
            color_to_column[color] = i

    # Place colors in output
    for color, col_idx in color_to_column.items():
        cross_count = color_info[color]['crosses']

        # Color appears in bottom row
        result[output_height - 1][col_idx] = color

        # If color has crosses, extend upward
        for i in range(cross_count):
            row_idx = output_height - 2 - i
            if row_idx >= 0:
                result[row_idx][col_idx] = color

    return result
