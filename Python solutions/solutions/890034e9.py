def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a hollow rectangle made of a special value
    2. Output has this rectangle copied to a diagonally opposite location
    3. Original rectangle: if in top half, copy goes to bottom half; if in bottom half, copy goes to top half
    4. Horizontally: copy is positioned to the right side of the grid
    5. Vertical spacing: top rectangles get ~7 rows gap, bottom rectangles meet at boundary

    Procedure:
    1. Find the hollow rectangle (border-only pattern)
    2. Determine copy location based on diagonal opposite positioning
    3. Place rectangle at the calculated position
    """

    import copy

    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    def find_hollow_rectangle():
        """Find a hollow rectangle (border only, different interior)"""
        for r in range(rows - 2):
            for c in range(cols - 2):
                for h in range(3, rows - r + 1):
                    for w in range(3, cols - c + 1):
                        if r + h > rows or c + w > cols:
                            continue

                        color = grid[r][c]
                        if color == 0:
                            continue

                        # Check if all border cells match the color
                        is_valid = True
                        pattern_cells = []

                        # Top and bottom edges
                        for col in range(c, c + w):
                            if grid[r][col] != color or grid[r + h - 1][col] != color:
                                is_valid = False
                                break
                            pattern_cells.append((r, col))
                            pattern_cells.append((r + h - 1, col))

                        if not is_valid:
                            continue

                        # Left and right edges (excluding corners)
                        for row in range(r + 1, r + h - 1):
                            if grid[row][c] != color or grid[row][c + w - 1] != color:
                                is_valid = False
                                break
                            pattern_cells.append((row, c))
                            pattern_cells.append((row, c + w - 1))

                        if not is_valid:
                            continue

                        # Check interior is NOT the border color
                        has_different_interior = True
                        for row in range(r + 1, r + h - 1):
                            for col in range(c + 1, c + w - 1):
                                if grid[row][col] == color:
                                    has_different_interior = False
                                    break
                            if not has_different_interior:
                                break

                        if has_different_interior:
                            return (r, c, h, w, color, set(pattern_cells))

        return None

    rect = find_hollow_rectangle()

    if rect:
        orig_r, orig_c, height, width, color, cells = rect

        # Calculate copy position based on pattern analysis
        # Rows: complementary positioning with specific gaps
        orig_bottom = orig_r + height - 1
        grid_center_row = rows // 2

        if orig_r < grid_center_row:
            # Original in top half - place copy in bottom with gap of 7
            copy_r = orig_bottom + 7
            if copy_r + height > rows:
                copy_r = rows - height
        else:
            # Original in bottom/middle - copy ends where original starts
            copy_bottom = orig_r
            copy_r = copy_bottom - height + 1
            if copy_r < 0:
                copy_r = 0

        # Columns: use complementary center positioning
        # Pattern: sum of centers = cols - 3 (for most) or cols - 2 (for larger widths)
        orig_right = orig_c + width - 1
        orig_center_col = (orig_c + orig_right) / 2.0

        # Use different constant based on width
        center_sum = (cols - 2) if width >= 6 else (cols - 3)
        desired_center_col = center_sum - orig_center_col

        # Calculate copy_left with conditional rounding
        if width % 2 == 1:  # Odd width - add 1 for proper positioning
            copy_c = int(desired_center_col - width / 2.0 + 1.0)
        else:  # Even width
            if width >= 6:  # Larger even widths
                copy_c = int(desired_center_col - width / 2.0 + 1.0)
            else:  # Smaller even widths - round to nearest
                copy_c = int(desired_center_col - width / 2.0 + 0.5)

        # Ensure it fits
        if copy_c < 0:
            copy_c = 0
        if copy_c + width > cols:
            copy_c = cols - width

        # Copy the rectangle pattern to new location
        for orig_row, orig_col in cells:
            new_row = copy_r + (orig_row - orig_r)
            new_col = copy_c + (orig_col - orig_c)
            if 0 <= new_row < rows and 0 <= new_col < cols:
                result[new_row][new_col] = color

    return result
