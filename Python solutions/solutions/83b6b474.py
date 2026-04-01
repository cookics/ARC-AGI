def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has background and non-background colored shapes
    2. Output is a square with non-background values on perimeter, background in center
    3. Output size determined by: perimeter = 4*s - 4 = count of non-background cells
    4. Example 1: 12 non-bg → 4×4 output, Example 2: 20 non-bg → 6×6 output
    5. Non-background cells from input bbox are rotated 90° clockwise, then read column-wise in reverse
    6. These values are placed clockwise around the output perimeter

    Procedure:
    1. Find background color (most frequent)
    2. Extract bounding box of non-background cells
    3. Rotate bbox 90° clockwise
    4. Read non-background values column-by-column from right to left
    5. Create output square and place values on perimeter clockwise
    6. Fill interior with background
    """

    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Find background (most frequent)
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_values).most_common(1)[0][0]

    # Find bounding box
    non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != background]

    if not non_bg:
        return grid

    min_r = min(r for r, c in non_bg)
    max_r = max(r for r, c in non_bg)
    min_c = min(c for r, c in non_bg)
    max_c = max(c for r, c in non_bg)

    # Extract bounding box
    bbox = []
    for r in range(min_r, max_r + 1):
        row = []
        for c in range(min_c, max_c + 1):
            row.append(grid[r][c])
        bbox.append(row)

    # Rotate 90° counter-clockwise: (r, c) → (cols-1-c, r)
    bbox_h, bbox_w = len(bbox), len(bbox[0])
    rotated = [[bbox[r][bbox_w - 1 - c] for r in range(bbox_h)] for c in range(bbox_w)]

    # Read non-background values column-by-column from left to right
    perimeter_values = []
    rotated_h, rotated_w = len(rotated), len(rotated[0])
    for c in range(rotated_w):
        for r in range(rotated_h):
            if rotated[r][c] != background:
                perimeter_values.append(rotated[r][c])

    # Reorder: keep first 'size', then swap the two middle quarters
    num_non_bg = len(perimeter_values)
    if num_non_bg < 4:
        return grid

    size = (num_non_bg + 4) // 4
    if num_non_bg >= 2 * size:
        perimeter_values = perimeter_values[:size] + perimeter_values[-size:] + perimeter_values[size:-size]

    # Create output grid

    size = (num_non_bg + 4) // 4
    result = [[background] * size for _ in range(size)]

    # Generate perimeter positions counter-clockwise from top-left
    perimeter_positions = []
    # Top row (left to right)
    for c in range(size):
        perimeter_positions.append((0, c))
    # Left column (top to bottom, excluding top)
    for r in range(1, size):
        perimeter_positions.append((r, 0))
    # Bottom row (left to right, excluding left)
    for c in range(1, size):
        perimeter_positions.append((size - 1, c))
    # Right column (bottom to top, excluding bottom and top)
    for r in range(size - 2, 0, -1):
        perimeter_positions.append((r, size - 1))

    # Place values on perimeter
    for i, val in enumerate(perimeter_values):
        if i < len(perimeter_positions):
            r, c = perimeter_positions[i]
            result[r][c] = val

    return result
