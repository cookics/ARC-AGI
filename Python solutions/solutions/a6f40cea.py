def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains multiple hollow rectangles of different colors
    2. Output is the interior of the largest hollow rectangle
    3. Other rectangles' frames are drawn onto this interior canvas
    4. Drawing uses the actual color values from input (to preserve patterns)

    Procedure:
    1. Find background color (most frequent)
    2. Detect all hollow rectangles by finding bounding boxes of each color
    3. Identify the largest rectangle by interior area
    4. Extract that rectangle's interior as the base output
    5. Draw all other rectangles' frames onto the output
    """

    from collections import Counter

    h, w = len(grid), len(grid[0])

    # Find background
    flat = [grid[r][c] for r in range(h) for c in range(w)]
    background = Counter(flat).most_common(1)[0][0]

    # Find rectangles for each non-background color
    rectangles = []
    colors = set(flat) - {background}

    for color in colors:
        # Find all cells of this color
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == color]
        if len(cells) < 8:  # Too few cells to form a meaningful rectangle
            continue

        # Get bounding box
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Check if this could be a hollow rectangle (has interior space)
        if max_r - min_r < 2 or max_c - min_c < 2:
            continue

        # Count how many border cells have this color
        border_count = 0
        total_border = 0
        for c in range(min_c, max_c + 1):
            total_border += 1
            if grid[min_r][c] == color:
                border_count += 1
            total_border += 1
            if grid[max_r][c] == color:
                border_count += 1
        for r in range(min_r + 1, max_r):
            total_border += 1
            if grid[r][min_c] == color:
                border_count += 1
            total_border += 1
            if grid[r][max_c] == color:
                border_count += 1

        # If at least 30% of border is this color, consider it a rectangle
        if border_count >= total_border * 0.3:
            interior_area = (max_r - min_r - 1) * (max_c - min_c - 1)
            if interior_area > 0:
                rectangles.append({
                    'color': color,
                    'min_r': min_r,
                    'max_r': max_r,
                    'min_c': min_c,
                    'max_c': max_c,
                    'area': interior_area
                })

    if not rectangles:
        return grid

    # Find largest rectangle
    largest = max(rectangles, key=lambda x: x['area'])

    # Extract interior of largest
    int_min_r = largest['min_r'] + 1
    int_max_r = largest['max_r']
    int_min_c = largest['min_c'] + 1
    int_max_c = largest['max_c']

    result = []
    for r in range(int_min_r, int_max_r):
        row = []
        for c in range(int_min_c, int_max_c):
            row.append(grid[r][c])
        result.append(row)

    out_h = len(result)
    out_w = len(result[0])

    # Draw other rectangles
    for rect in rectangles:
        if rect == largest:
            continue

        r1, r2 = rect['min_r'], rect['max_r']
        c1, c2 = rect['min_c'], rect['max_c']
        color = rect['color']

        # Draw hollow rectangle frame
        # For each edge position, draw with the appropriate color
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                # Check if on frame
                is_frame = (r == r1 or r == r2 or c == c1 or c == c2)
                if not is_frame:
                    continue

                # Transform to output coordinates
                out_r = r - int_min_r
                out_c = c - int_min_c

                # Check bounds
                if 0 <= out_r < out_h and 0 <= out_c < out_w:
                    # If input cell has non-background color, use it (preserves patterns)
                    # Otherwise, use the rectangle's primary color
                    input_val = grid[r][c]
                    if input_val != background:
                        result[out_r][out_c] = input_val
                    else:
                        result[out_r][out_c] = color

    return result
