def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains L-shaped patterns (3 cells forming an L) scattered in the grid
    2. Bottom-left corner has a legend with color mappings [L-shape color, fill color]
    3. If single legend entry: create frame around ALL L-shapes
    4. If multiple legend entries: fill bounding box per color separately

    Procedure:
    1. Extract legend from bottom rows
    2. Identify all L-shapes in the grid (excluding legend area)
    3. If single legend entry, create frame around all L-shapes
    4. If multiple entries, fill bounding box per color
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Identify background color
    from collections import Counter
    all_vals = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg = Counter(all_vals).most_common(1)[0][0]

    # Extract legend from bottom-left corner (last 5 rows, first 4 columns)
    legend = {}  # L-shape color -> fill color
    legend_area = set()
    legend_min_row = rows

    for r in range(max(0, rows - 5), rows):
        for c in range(min(4, cols) - 1):
            if c + 1 < cols:
                val1, val2 = grid[r][c], grid[r][c + 1]
                if val1 != bg and val2 != bg:
                    legend[val1] = val2
                    legend_area.add((r, c))
                    legend_area.add((r, c + 1))
                    legend_min_row = min(legend_min_row, r)

    # Find all L-shapes (excluding legend area)
    def find_l_shapes():
        shapes = {}
        used = set()

        for r in range(rows):
            for c in range(cols):
                if (r, c) in used or (r, c) in legend_area:
                    continue

                # Try all 4 L-shape orientations
                # Each pattern is defined by its 3 cells
                patterns = [
                    [(r, c), (r, c + 1), (r + 1, c)],      # ┐
                    [(r, c), (r, c + 1), (r + 1, c + 1)],  # ┌
                    [(r, c), (r + 1, c), (r + 1, c + 1)],  # └
                    [(r, c + 1), (r + 1, c), (r + 1, c + 1)],  # ┘
                ]

                for cells in patterns:
                    # Check bounds
                    if not all(0 <= rr < rows and 0 <= cc < cols for rr, cc in cells):
                        continue
                    # Check not in legend
                    if any((rr, cc) in legend_area for rr, cc in cells):
                        continue
                    # Check not already used
                    if any((rr, cc) in used for rr, cc in cells):
                        continue

                    # Get color from first cell
                    color = grid[cells[0][0]][cells[0][1]]
                    if color == bg:
                        continue

                    # Check all cells have same color
                    if all(grid[rr][cc] == color for rr, cc in cells):
                        if color not in shapes:
                            shapes[color] = []
                        shapes[color].append(cells)
                        for rr, cc in cells:
                            used.add((rr, cc))
                        break

        return shapes

    l_shapes = find_l_shapes()

    # Collect all L-shape cells to preserve them
    l_shape_cells = set()
    for color_shapes in l_shapes.values():
        for l_shape in color_shapes:
            l_shape_cells.update(l_shape)

    # Identify colors that should be treated as removable (fill_color == 0)
    removable_colors = {color for color, fill in legend.items() if fill == 0}

    # Different fill strategy based on number of legend entries
    if len(legend) == 1:
        # Single entry: create thick frame around ALL L-shapes
        fill_color = list(legend.values())[0]
        all_l_cells = []
        for color_shapes in l_shapes.values():
            for l_shape in color_shapes:
                all_l_cells.extend(l_shape)

        if all_l_cells:
            min_r = min(r for r, c in all_l_cells)
            max_r = max(r for r, c in all_l_cells)
            min_c = min(c for r, c in all_l_cells)
            max_c = max(c for r, c in all_l_cells)

            # Create frame with thicker top/bottom borders
            for r in range(min_r, max_r + 1):
                for c in range(min_c, max_c + 1):
                    if result[r][c] == bg and (r, c) not in legend_area:
                        # Fill left/right borders
                        if c == min_c or c == max_c:
                            result[r][c] = fill_color
                        # Fill top 2 rows completely
                        elif r == min_r or r == min_r + 1:
                            result[r][c] = fill_color
                        # Fill bottom 2 rows completely
                        elif r == max_r or r == max_r - 1:
                            result[r][c] = fill_color
    else:
        # Multiple entries: find ALL cells of each color and fill bounding box
        for color, fill_color in legend.items():
            # Skip if fill_color is 0 (means ignore this color)
            if fill_color == 0:
                continue

            # Find all cells of this color (not just L-shapes)
            color_cells = []
            for r in range(rows):
                for c in range(cols):
                    if (r, c) not in legend_area and grid[r][c] == color:
                        color_cells.append((r, c))

            if color_cells:
                min_r = min(r for r, c in color_cells)
                max_r = max(r for r, c in color_cells)
                min_c = min(c for r, c in color_cells)
                max_c = max(c for r, c in color_cells)

                # Fill bounding box
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        if (r, c) in legend_area:
                            continue
                        # Preserve L-shapes of OTHER colors (not source, not removable)
                        if (r, c) in l_shape_cells and grid[r][c] != color and grid[r][c] not in removable_colors:
                            continue
                        # Fill background, all source color cells, or removable cells
                        if result[r][c] == bg or grid[r][c] == color or grid[r][c] in removable_colors:
                            result[r][c] = fill_color

    return result
