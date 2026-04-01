def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a repeating tile pattern
    2. Some rectangular regions are filled with constant values (occlusions)
    3. Output is the reconstruction of the largest occluded rectangular region

    Procedure:
    1. Find all rectangular regions filled with single values
    2. Select the largest rectangular region by area
    3. Detect the repeating pattern using non-occluded cells
    4. Reconstruct what should be in the largest occluded region
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])

    # Find all rectangular regions for each unique value
    def find_largest_rect_for_value(value):
        """Find the largest rectangular region filled entirely with 'value'"""
        best_rect = None
        best_area = 0

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == value:
                    # Find max width starting from this cell
                    max_width = 0
                    while c + max_width < cols and grid[r][c + max_width] == value:
                        max_width += 1

                    # Try expanding downward
                    for height in range(1, rows - r + 1):
                        # Check if row r+height-1 has 'value' from c to c+max_width-1
                        if r + height - 1 >= rows:
                            break

                        valid_width = 0
                        for w in range(max_width):
                            if grid[r + height - 1][c + w] == value:
                                valid_width += 1
                            else:
                                break

                        max_width = min(max_width, valid_width)
                        if max_width == 0:
                            break

                        area = height * max_width
                        if area > best_area:
                            best_area = area
                            best_rect = {
                                'r1': r, 'r2': r + height - 1,
                                'c1': c, 'c2': c + max_width - 1,
                                'area': area, 'value': value
                            }

        return best_rect

    # Find all unique values and their largest rectangles
    unique_values = set()
    for row in grid:
        unique_values.update(row)

    # Count occurrences of each value
    value_counts = {}
    total_cells = rows * cols
    for row in grid:
        for val in row:
            value_counts[val] = value_counts.get(val, 0) + 1

    # Find the most common value (likely part of the pattern, not an occlusion)
    most_common_value = max(value_counts, key=value_counts.get)

    # Find all significant rectangles for non-dominant values
    all_rects = []
    for value in unique_values:
        if value == most_common_value:
            continue  # Skip the most common value
        rect = find_largest_rect_for_value(value)
        if rect and rect['area'] >= 9:  # Lower threshold to catch smaller occlusions
            all_rects.append(rect)
            # Debug: uncomment to see what rectangles are found
            print(f"Found rect for value {value}: {rect}")

    if not all_rects:
        # Debug: uncomment to see why no rects were found
        print(f"No rects found. Most common: {most_common_value}, counts: {value_counts}")
        return grid

    # Sort by area and get top candidates
    all_rects.sort(key=lambda x: x['area'], reverse=True)

    # Detect pattern period using non-masked cells
    def detect_period(mask):
        for row_period in range(1, min(20, rows)):
            for col_period in range(1, min(20, cols)):
                # Build pattern from non-masked cells
                pattern = [[None] * col_period for _ in range(row_period)]
                valid = True

                for r in range(rows):
                    for c in range(cols):
                        if not mask[r][c]:
                            pr, pc = r % row_period, c % col_period
                            if pattern[pr][pc] is None:
                                pattern[pr][pc] = grid[r][c]
                            elif pattern[pr][pc] != grid[r][c]:
                                valid = False
                                break
                    if not valid:
                        break

                # Check all pattern cells are filled
                if valid:
                    all_filled = True
                    for row in pattern:
                        if None in row:
                            all_filled = False
                            break
                    if all_filled:
                        return pattern, row_period, col_period

        return None, 0, 0

    # Try masking progressively more rectangles until pattern detection works
    pattern, row_period, col_period = None, 0, 0
    mask = [[False] * cols for _ in range(rows)]

    for num_to_mask in range(1, min(len(all_rects) + 1, 6)):
        mask = [[False] * cols for _ in range(rows)]
        for rect in all_rects[:num_to_mask]:
            for r in range(rect['r1'], rect['r2'] + 1):
                for c in range(rect['c1'], rect['c2'] + 1):
                    mask[r][c] = True

        # Try to detect pattern with this masking
        pattern, row_period, col_period = detect_period(mask)
        print(f"Tried masking {num_to_mask} rects, pattern found: {pattern is not None}, period: {row_period}x{col_period}")
        if pattern is not None:
            break

    if pattern is None:
        print("No pattern detected!")
        return grid

    # Select the largest rectangle
    target = all_rects[0]

    # Reconstruct the target region
    result = []
    for r in range(target['r1'], target['r2'] + 1):
        row = []
        for c in range(target['c1'], target['c2'] + 1):
            pr, pc = r % row_period, c % col_period
            row.append(pattern[pr][pc])
        result.append(row)

    return result
