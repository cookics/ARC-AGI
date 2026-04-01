def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background color (1) and several other colors scattered around
    2. One color forms a complete shape (the template) - typically the one with most cells
    3. Other colors have partial/scattered cells that are subsets of the template shape
    4. Output completes all partial shapes to match the template shape

    Procedure:
    1. Find all colors in the grid (excluding background color 1)
    2. For each color, collect all cell positions
    3. Identify the template: the color with the most cells (most complete shape)
    4. Convert template to relative positions (normalized to top-left corner)
    5. For each other color (scattered/partial):
       - Find the offset where the template would fit the scattered cells
       - Fill in the complete template at that offset with the color
    """

    # Find all colors except background
    colors = set()
    for row in grid:
        colors.update(row)

    background = 1
    colors.discard(background)

    if not colors:
        return grid

    # Get cells for each color
    color_cells = {}
    for color in colors:
        cells = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == color:
                    cells.append((r, c))
        color_cells[color] = cells

    # Find the template (color with most cells = most complete)
    template_color = max(color_cells, key=lambda c: len(color_cells[c]))
    template_cells = color_cells[template_color]

    # Convert template to relative positions (normalized to bounding box top-left)
    min_r = min(r for r, c in template_cells)
    min_c = min(c for r, c in template_cells)
    template = set((r - min_r, c - min_c) for r, c in template_cells)

    # Create result grid (copy of input)
    result = [row[:] for row in grid]

    # Complete each scattered color to match template
    for color in colors:
        if color == template_color:
            continue

        cells = color_cells[color]
        if not cells:
            continue

        # Find offset where scattered cells align with template
        # Try mapping the first scattered cell to each template position
        first_cell = cells[0]

        for tr, tc in template:
            # Calculate offset if first_cell maps to template position (tr, tc)
            offset_r = first_cell[0] - tr
            offset_c = first_cell[1] - tc

            # Verify all scattered cells match some template position with this offset
            all_match = True
            for r, c in cells:
                rel_r, rel_c = r - offset_r, c - offset_c
                if (rel_r, rel_c) not in template:
                    all_match = False
                    break

            if all_match:
                # Fill in the complete template at this offset
                for tr2, tc2 in template:
                    abs_r, abs_c = offset_r + tr2, offset_c + tc2
                    if 0 <= abs_r < len(grid) and 0 <= abs_c < len(grid[0]):
                        result[abs_r][abs_c] = color
                break

    return result
