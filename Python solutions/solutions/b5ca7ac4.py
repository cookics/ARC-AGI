def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 5×5 hollow rectangles with uniform borders and interiors
    2. Border values are either 8 or 2, interiors have different values
    3. Rectangles with border=8 slide LEFT as far as possible
    4. Rectangles with border=2 slide RIGHT as far as possible
    5. Rectangles stop when they would collide with same-border-type rectangles in overlapping rows
    6. Background is the most common value in the grid

    Procedure:
    1. Identify background value (most frequent)
    2. Find all 5×5 rectangles by checking border and interior patterns
    3. Separate rectangles by border type (8 or 2)
    4. Process border=8 rectangles left-to-right: move each as far left as possible
    5. Process border=2 rectangles right-to-left: move each as far right as possible
    6. Reconstruct output grid with rectangles in their new positions
    """

    height = len(grid)
    width = len(grid[0])

    # Find background value (most common value)
    from collections import Counter
    flat = [cell for row in grid for cell in row]
    background = Counter(flat).most_common(1)[0][0]

    # Find all 5×5 rectangles
    rectangles = []

    for r in range(height - 4):
        for c in range(width - 4):
            border_val = grid[r][c]

            # Only consider borders with value 2 or 8
            if border_val not in [2, 8]:
                continue

            # Check if this forms a valid 5×5 rectangle
            is_valid = True

            # Check top and bottom borders (all should be border_val)
            for i in range(5):
                if grid[r][c+i] != border_val or grid[r+4][c+i] != border_val:
                    is_valid = False
                    break

            if not is_valid:
                continue

            # Check left and right borders (all should be border_val)
            for i in range(5):
                if grid[r+i][c] != border_val or grid[r+i][c+4] != border_val:
                    is_valid = False
                    break

            if not is_valid:
                continue

            # Check interior (3×3 region should be uniform and different from border)
            interior_val = grid[r+1][c+1]
            if interior_val == border_val:
                continue

            for i in range(1, 4):
                for j in range(1, 4):
                    if grid[r+i][c+j] != interior_val:
                        is_valid = False
                        break
                if not is_valid:
                    break

            if is_valid:
                # Check if this overlaps with already found rectangles
                overlaps = False
                for existing in rectangles:
                    r_overlap = not (existing['r'] + 5 <= r or r + 5 <= existing['r'])
                    c_overlap = not (existing['c'] + 5 <= c or c + 5 <= existing['c'])
                    if r_overlap and c_overlap:
                        overlaps = True
                        break

                if not overlaps:
                    rectangles.append({
                        'r': r,
                        'c': c,
                        'border': border_val,
                        'interior': interior_val
                    })

    # Helper function to check if two rectangles have overlapping rows
    def rows_overlap(r1, r2):
        return not (r1 + 5 <= r2 or r2 + 5 <= r1)

    # Process border=8 rectangles (move left)
    # Sort by column (process leftmost first)
    border_8 = [rect for rect in rectangles if rect['border'] == 8]
    border_8.sort(key=lambda x: x['c'])

    for i, rect in enumerate(border_8):
        target_c = 0  # Try to move to left edge

        # Check against already-processed rectangles with border=8
        for j in range(i):
            other = border_8[j]
            if rows_overlap(rect['r'], other['r']):
                # Must be to the right of this rectangle
                target_c = max(target_c, other['new_c'] + 5)

        rect['new_c'] = target_c

    # Process border=2 rectangles (move right)
    # Sort by column descending (process rightmost first)
    border_2 = [rect for rect in rectangles if rect['border'] == 2]
    border_2.sort(key=lambda x: -x['c'])

    for i, rect in enumerate(border_2):
        target_c = width - 5  # Try to move to right edge

        # Check against already-processed rectangles with border=2
        for j in range(i):
            other = border_2[j]
            if rows_overlap(rect['r'], other['r']):
                # Must be to the left of this rectangle
                target_c = min(target_c, other['new_c'] - 5)

        rect['new_c'] = target_c

    # Create output grid filled with background
    output = [[background] * width for _ in range(height)]

    # Draw all rectangles at their new positions
    for rect in rectangles:
        r = rect['r']
        c = rect['new_c']
        border_val = rect['border']
        interior_val = rect['interior']

        # Draw the 5×5 rectangle
        for i in range(5):
            for j in range(5):
                if i == 0 or i == 4 or j == 0 or j == 4:
                    # Border cells
                    output[r+i][c+j] = border_val
                else:
                    # Interior cells
                    output[r+i][c+j] = interior_val

    return output
