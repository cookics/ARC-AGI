def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains non-zero colored cells forming a shape (outline/pattern)
    2. Bounding box transformation is consistent: height' = height - 2, width' = width + 1, offset (+1, +1)
    3. Two different fill patterns:
       - L-shape/rectangular frame (Example 1): Fill entire rectangle
       - Diamond shape (Examples 2 & 3): Fill diamond pattern (2 cells at edges, 4 cells at center)
    4. Diamond detection: Max points per row/col <= 2 (diagonal pattern)
    5. Example 1: 5×3 bbox → 3×4 filled rectangle at (+1,+1)
       Example 2: 6×3 bbox → 4×4 filled diamond at (+1,+1)
       Example 3: 5×3 bbox → 3×4 filled diamond at (+1,+1)

    Procedure:
    1. Find all non-zero cells, compute bounding box and color
    2. Detect shape type: diamond (diagonal) vs rectangular frame
    3. Calculate new bbox: height' = height - 2, width' = width + 1, shift (+1, +1)
    4. Fill accordingly:
       - Rectangle: Fill all cells in new bbox
       - Diamond: Fill with pattern based on distance from center row
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all non-zero points and identify color
    points = []
    color = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                points.append((r, c))
                color = grid[r][c]

    if not points:
        return result

    # Find bounding box of non-zero cells
    min_r = min(p[0] for p in points)
    max_r = max(p[0] for p in points)
    min_c = min(p[1] for p in points)
    max_c = max(p[1] for p in points)

    # Calculate original dimensions
    old_height = max_r - min_r + 1
    old_width = max_c - min_c + 1

    # Calculate new dimensions
    new_height = old_height - 2
    new_width = old_width + 1

    # Calculate new position (shift by +1, +1)
    new_min_r = min_r + 1
    new_min_c = min_c + 1
    new_max_r = new_min_r + new_height - 1
    new_max_c = new_min_c + new_width - 1

    # Detect shape type: diamond if max points per row/col <= 2 (diagonal pattern)
    row_counts = {}
    col_counts = {}
    for r, c in points:
        row_counts[r] = row_counts.get(r, 0) + 1
        col_counts[c] = col_counts.get(c, 0) + 1

    max_per_row = max(row_counts.values())
    max_per_col = max(col_counts.values())
    is_diamond = max_per_row <= 2 and max_per_col <= 2

    if is_diamond:
        # Fill diamond pattern
        center_r = (new_min_r + new_max_r) / 2
        center_c = (new_min_c + new_max_c) / 2

        for r in range(new_min_r, new_max_r + 1):
            dist = abs(r - center_r)
            if dist >= 1:
                # Edge rows: 2 cells wide, centered
                left_c = int(center_c)
                right_c = int(center_c) + 1
            else:
                # Center rows: full width
                left_c = new_min_c
                right_c = new_max_c

            for c in range(left_c, right_c + 1):
                if c < cols:
                    result[r][c] = color
    else:
        # Fill rectangle
        for r in range(new_min_r, new_max_r + 1):
            for c in range(new_min_c, new_max_c + 1):
                if r < rows and c < cols:
                    result[r][c] = color

    return result
