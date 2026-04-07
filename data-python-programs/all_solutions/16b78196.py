def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a band (horizontal or vertical) of one dominant color with holes
    2. Small colored shapes scattered on both sides of the band
    3. Shapes are collected, merged by color, and stacked at hole positions
    4. Shapes from one side of band move to opposite side
    5. Shapes are overlaid to create combined patterns

    Procedure:
    1. Find the dominant band and its holes
    2. Extract all colored shapes by color (merge same-color components)
    3. Group shapes by which side of band they're on
    4. Stack shapes from each side at holes on opposite side
    5. Overlay with transparency (non-zero wins over zero)
    """

    from collections import deque, defaultdict, Counter

    height, width = len(grid), len(grid[0])

    # Find horizontal band
    def find_horiz_band():
        for i in range(height):
            row_colors = [grid[i][j] for j in range(width) if grid[i][j] != 0]
            if len(row_colors) > width * 0.75:
                counter = Counter(row_colors)
                band_color, count = counter.most_common(1)[0]
                if count > width * 0.75:
                    # Expand to find full band extent
                    start, end = i, i
                    while start > 0 and any(grid[start-1][j] == band_color for j in range(width)):
                        start -= 1
                    while end < height-1 and any(grid[end+1][j] == band_color for j in range(width)):
                        end += 1
                    return ('horizontal', start, end, band_color)
        return None

    # Find vertical band
    def find_vert_band():
        for j in range(width):
            col_colors = [grid[i][j] for i in range(height) if grid[i][j] != 0]
            if len(col_colors) > height * 0.75:
                counter = Counter(col_colors)
                band_color, count = counter.most_common(1)[0]
                if count > height * 0.75:
                    # Expand to find full band extent
                    start, end = j, j
                    while start > 0 and any(grid[i][start-1] == band_color for i in range(height)):
                        start -= 1
                    while end < width-1 and any(grid[i][end+1] == band_color for i in range(height)):
                        end += 1
                    return ('vertical', start, end, band_color)
        return None

    band = find_horiz_band()
    if not band:
        band = find_vert_band()
    if not band:
        return grid

    band_type, band_start, band_end, band_color = band

    # Extract all cells by color (merging same-color components)
    color_shapes = defaultdict(list)
    for i in range(height):
        for j in range(width):
            if grid[i][j] != 0 and grid[i][j] != band_color:
                color_shapes[grid[i][j]].append((i, j))

    # Initialize result with zeros
    result = [[0] * width for _ in range(height)]

    # Copy band to result
    for i in range(height):
        for j in range(width):
            if grid[i][j] == band_color:
                result[i][j] = band_color

    if band_type == 'horizontal':
        band_mid = (band_start + band_end) // 2

        # Separate shapes by side
        above_colors, below_colors = {}, {}
        for color, cells in color_shapes.items():
            avg_row = sum(r for r, c in cells) / len(cells)
            if avg_row < band_mid:
                above_colors[color] = cells
            else:
                below_colors[color] = cells

        # Sort colors by size (number of cells)
        all_colors = list(color_shapes.keys())
        all_colors.sort(key=lambda c: len(color_shapes[c]))

        #  Split into two groups
        mid = len(all_colors) // 2
        group1 = all_colors[:mid+1] if len(all_colors) % 2 else all_colors[:mid]
        group2 = all_colors[mid+1:] if len(all_colors) % 2 else all_colors[mid:]

        # Find holes in band (positions with 0 in the band rows)
        holes = []
        for j in range(width):
            if any(grid[i][j] == 0 for i in range(band_start, band_end+1)):
                if not holes or j > holes[-1][1] + 1:
                    holes.append([j, j])
                else:
                    holes[-1][1] = j

        # Stack group1 above band at first hole
        if holes and group1:
            hole_col = (holes[0][0] + holes[0][1]) // 2
            stack_row = band_start - 1
            for color in reversed(group1):
                cells = color_shapes[color]
                min_r, max_r = min(r for r, c in cells), max(r for r, c in cells)
                min_c, max_c = min(c for r, c in cells), max(c for r, c in cells)

                # Center shape at hole column
                col_offset = hole_col - (min_c + max_c) // 2

                for r, c in cells:
                    new_r = stack_row - (max_r - r)
                    new_c = c + col_offset
                    if 0 <= new_r < height and 0 <= new_c < width:
                        if result[new_r][new_c] == 0:
                            result[new_r][new_c] = color

                stack_row -= (max_r - min_r + 1)

        # Stack group2 below band at second hole (or first if only one)
        if holes and group2:
            hole_idx = 1 if len(holes) > 1 else 0
            hole_col = (holes[hole_idx][0] + holes[hole_idx][1]) // 2
            stack_row = band_end + 1
            for color in group2:
                cells = color_shapes[color]
                min_r, max_r = min(r for r, c in cells), max(r for r, c in cells)
                min_c, max_c = min(c for r, c in cells), max(c for r, c in cells)

                # Center shape at hole column
                col_offset = hole_col - (min_c + max_c) // 2

                for r, c in cells:
                    new_r = stack_row + (r - min_r)
                    new_c = c + col_offset
                    if 0 <= new_r < height and 0 <= new_c < width:
                        if result[new_r][new_c] == 0:
                            result[new_r][new_c] = color

                stack_row += (max_r - min_r + 1)

    else:  # vertical band
        band_mid = (band_start + band_end) // 2

        # Separate shapes by side
        left_colors, right_colors = {}, {}
        for color, cells in color_shapes.items():
            avg_col = sum(c for r, c in cells) / len(cells)
            if avg_col < band_mid:
                left_colors[color] = cells
            else:
                right_colors[color] = cells

        # Sort colors by size
        all_colors = list(color_shapes.keys())
        all_colors.sort(key=lambda c: len(color_shapes[c]))

        # Split into two groups
        mid = len(all_colors) // 2
        group1 = all_colors[:mid+1] if len(all_colors) % 2 else all_colors[:mid]
        group2 = all_colors[mid+1:] if len(all_colors) % 2 else all_colors[mid:]

        # Find holes in band
        holes = []
        for i in range(height):
            if any(grid[i][j] == 0 for j in range(band_start, band_end+1)):
                if not holes or i > holes[-1][1] + 1:
                    holes.append([i, i])
                else:
                    holes[-1][1] = i

        # Stack group1 left of band at first hole
        if holes and group1:
            hole_row = (holes[0][0] + holes[0][1]) // 2
            stack_col = band_start - 1
            for color in reversed(group1):
                cells = color_shapes[color]
                min_r, max_r = min(r for r, c in cells), max(r for r, c in cells)
                min_c, max_c = min(c for r, c in cells), max(c for r, c in cells)

                # Center shape at hole row
                row_offset = hole_row - (min_r + max_r) // 2

                for r, c in cells:
                    new_r = r + row_offset
                    new_c = stack_col - (max_c - c)
                    if 0 <= new_r < height and 0 <= new_c < width:
                        if result[new_r][new_c] == 0:
                            result[new_r][new_c] = color

                stack_col -= (max_c - min_c + 1)

        # Stack group2 right of band at second hole
        if holes and group2:
            hole_idx = 1 if len(holes) > 1 else 0
            hole_row = (holes[hole_idx][0] + holes[hole_idx][1]) // 2
            stack_col = band_end + 1
            for color in group2:
                cells = color_shapes[color]
                min_r, max_r = min(r for r, c in cells), max(r for r, c in cells)
                min_c, max_c = min(c for r, c in cells), max(c for r, c in cells)

                # Center shape at hole row
                row_offset = hole_row - (min_r + max_r) // 2

                for r, c in cells:
                    new_r = r + row_offset
                    new_c = stack_col + (c - min_c)
                    if 0 <= new_r < height and 0 <= new_c < width:
                        if result[new_r][new_c] == 0:
                            result[new_r][new_c] = color

                stack_col += (max_c - min_c + 1)

    return result
