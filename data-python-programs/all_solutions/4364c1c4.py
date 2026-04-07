def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a dominant background color and several distinct colored regions
    2. Output has the same colored regions but shifted horizontally by 1 column
    3. Shift direction alternates based on vertical appearance order: 1st region left, 2nd right, 3rd left, etc.
    4. Cells that shift out of bounds are discarded and replaced with background color

    Procedure:
    1. Identify the background color (most frequent color in the grid)
    2. Find all distinct non-background colors and record their first appearance row
    3. Sort colors by their first appearance row (vertical order from top to bottom)
    4. For each color in order: shift left if even index (0,2,4...), shift right if odd index (1,3,5...)
    5. Clear original positions and place shifted cells within bounds
    """

    rows, cols = len(grid), len(grid[0])

    # Find background color (most frequent)
    color_count = {}
    for row in grid:
        for cell in row:
            color_count[cell] = color_count.get(cell, 0) + 1

    background = max(color_count, key=color_count.get)

    # Find all colored regions and their first appearance row
    colored_regions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in colored_regions:
                    colored_regions[color] = r

    # Sort regions by their first appearance (vertical order)
    region_order = sorted(colored_regions.items(), key=lambda x: x[1])

    # Create output grid as copy of input
    result = [row[:] for row in grid]

    # Apply shifts to each region
    for i, (color, _) in enumerate(region_order):
        shift_left = (
            i % 2 == 0
        )  # First region (index 0) shifts left, second (index 1) shifts right, etc.

        # Find all cells of this color
        color_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    color_cells.append((r, c))

        # Clear the original positions
        for r, c in color_cells:
            result[r][c] = background

        # Apply shift and place in new positions
        for r, c in color_cells:
            if shift_left:
                new_c = c - 1
            else:
                new_c = c + 1

            # Only place if within bounds
            if 0 <= new_c < cols:
                result[r][new_c] = color

    return result
