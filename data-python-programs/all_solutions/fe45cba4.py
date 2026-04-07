def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a background color (7) and two other colors (e.g., 9 and 2/8)
    2. The grid is divided horizontally: top region, separator row, bottom region
    3. One non-background color belongs to the top region, another to the bottom region
    4. Separator row (at height//2 - 1) becomes all background color
    5. Each non-background color is either preserved (if on right side only) or rearranged into rectangle
    6. "Scattered" means color has cells in left half of its region
    7. When rearranged, color forms a rectangle in the right half of its region

    Procedure:
    1. Identify separator row at position (height // 2) - 1
    2. Define top region (rows 0 to separator-1) and bottom region (rows separator+1 to end)
    3. Find all non-background colors
    4. For each color, determine its region based on average row position
    5. Check if color is scattered (has cells in left half of region)
    6. If not scattered, preserve color's positions
    7. If scattered, rearrange into rectangle in right half of region
    """

    height = len(grid)
    width = len(grid[0])

    # Background color (7 in all examples)
    bg_color = 7

    # Separator row
    separator_row = (height // 2) - 1

    # Define regions
    top_region_rows = list(range(0, separator_row))
    bottom_region_rows = list(range(separator_row + 1, height))

    # Find all non-background colors
    colors = set()
    for row in grid:
        for cell in row:
            if cell != bg_color:
                colors.add(cell)

    # Analyze each color
    color_info = {}
    for color in colors:
        # Find all cells of this color
        all_cells = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == color]

        # Determine region based on average row position
        avg_row = sum(r for r, c in all_cells) / len(all_cells)
        region = 'top' if avg_row <= separator_row else 'bottom'

        # Get cells in the appropriate region
        if region == 'top':
            region_cells = [(r, c) for r, c in all_cells if r in top_region_rows]
        else:
            region_cells = [(r, c) for r, c in all_cells if r in bottom_region_rows]

        # Check if scattered (has cells in left half of region)
        left_half_bound = width // 2
        has_left = any(c < left_half_bound for r, c in region_cells)

        color_info[color] = {
            'region': region,
            'scattered': has_left,
            'region_cells': region_cells,
            'total_count': len(all_cells)
        }

    # Create output grid filled with background color
    output = [[bg_color] * width for _ in range(height)]

    # Place each color in the output
    for color, info in color_info.items():
        if not info['scattered']:
            # Preserve as-is
            for r, c in info['region_cells']:
                output[r][c] = color
        else:
            # Rearrange into rectangle in right half
            count = info['total_count']
            rect_width = width // 2
            rect_height = (count + rect_width - 1) // rect_width

            # Starting position
            if info['region'] == 'top':
                start_row = 0
            else:
                start_row = separator_row + 1
            start_col = width // 2

            # Fill rectangle
            filled = 0
            for r in range(start_row, min(start_row + rect_height, height)):
                for c in range(start_col, width):
                    if filled < count:
                        output[r][c] = color
                        filled += 1

    return output
