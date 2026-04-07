def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 3 colored points forming a diagonal line
    2. Down-left diagonal: nested rectangles from outer grid edges inward
    3. Down-right diagonal: expanding rectangles from diagonal outward
    4. Vertical edges start from next grid row, not immediately after top

    Procedure:
    1. Find points, spacing, and direction
    2. Generate grid rows/cols based on point positions
    3. Draw hollow nested rectangles level by level
    """

    rows, cols = len(grid), len(grid[0])

    # Find colored points
    points = []
    color = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                points.append((r, c))
                color = grid[r][c]

    if len(points) < 3:
        return grid

    points.sort()
    spacing = points[1][0] - points[0][0]
    dc = points[1][1] - points[0][1]
    is_down_right = dc > 0

    result = [[0] * cols for _ in range(rows)]

    min_row, max_row = points[0][0], points[-1][0]
    min_col = min(p[1] for p in points)
    max_col = max(p[1] for p in points)

    # Generate grid positions
    grid_rows = []
    r = min_row % spacing
    while r < rows:
        grid_rows.append(r)
        r += spacing

    grid_cols = []
    c_start = (min_col if is_down_right else max_col) % spacing
    c = c_start
    while c < cols:
        grid_cols.append(c)
        c += spacing

    if is_down_right:
        # Down-right: expanding rectangles outward from diagonal
        min_row_idx = grid_rows.index(min_row)
        max_row_idx = grid_rows.index(max_row)
        min_col_idx = grid_cols.index(min_col)
        max_col_idx = grid_cols.index(max_col)

        max_expansion = max(min_row_idx, min_col_idx,
                            len(grid_rows) - 1 - max_row_idx,
                            len(grid_cols) - 1 - max_col_idx) + 1

        for exp in range(max_expansion):
            top_idx = max(0, min_row_idx - exp)
            bottom_idx = min(len(grid_rows) - 1, max_row_idx + exp)
            left_idx = max(0, min_col_idx - exp)
            right_idx = min(len(grid_cols) - 1, max_col_idx + exp)

            top, bottom = grid_rows[top_idx], grid_rows[bottom_idx]
            left, right = grid_cols[left_idx], grid_cols[right_idx]

            # Horizontal edges
            for c in range(left, right + 1):
                result[top][c] = color
                if bottom != top:
                    result[bottom][c] = color

            # Vertical edges - start from next grid row after top
            start_r = grid_rows[top_idx + 1] if top_idx + 1 <= bottom_idx else bottom
            for r in range(start_r, bottom):
                result[r][left] = color
                if right != left:
                    result[r][right] = color

    else:
        # Down-left: nested rectangles from outer edges
        n_levels = (min(len(grid_rows), len(grid_cols)) + 1) // 2

        for level in range(n_levels):
            top_idx = level
            bottom_idx = len(grid_rows) - 1 - level
            left_idx = level
            right_idx = len(grid_cols) - 1 - level

            if top_idx > bottom_idx or left_idx > right_idx:
                break

            top, bottom = grid_rows[top_idx], grid_rows[bottom_idx]
            left, right = grid_cols[left_idx], grid_cols[right_idx]

            # Top edge spans parent's boundaries (or full width for level 0)
            if level == 0 and grid_rows[0] > 0:
                # Outermost level with grid starting > 0: full width
                for c in range(cols):
                    result[top][c] = color
            elif level > 0:
                # Nested levels: span from parent's left to parent's right
                parent_left = grid_cols[left_idx - 1]
                parent_right = grid_cols[right_idx + 1]
                for c in range(parent_left, parent_right + 1):
                    result[top][c] = color
            else:
                # Level 0 starting at row 0: use own boundaries
                for c in range(left, right + 1):
                    result[top][c] = color

            # Bottom edge uses own boundaries
            if bottom != top:
                for c in range(left, right + 1):
                    result[bottom][c] = color

            # Vertical edges - start from next grid row after top
            start_r = grid_rows[top_idx + 1] if top_idx + 1 <= bottom_idx else bottom
            for r in range(start_r, bottom):
                result[r][left] = color
                if right != left:
                    result[r][right] = color

    return result
