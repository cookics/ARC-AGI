def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a separator line (row or column of all 1s)
    2. Templates on one side, 5s on the other
    3. Each row/col of 5s maps to the template in the corresponding row/col
    4. If vertical separator: same row; if horizontal separator: same column

    Procedure:
    1. Find separator
    2. For each row (vertical sep) or column (horizontal sep), find dominant template color
    3. Replace 5s in that row/col with the template color
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find separator (check for lines with mostly 1s)
    sep_row = None
    sep_col = None

    # Check for horizontal separator (row with all or mostly 1s)
    for r in range(rows):
        ones_count = sum(1 for c in range(cols) if grid[r][c] == 1)
        if ones_count >= cols * 0.9:  # 90% or more are 1s
            sep_row = r
            break

    # Check for vertical separator (column with all or mostly 1s)
    for c in range(cols):
        ones_count = sum(1 for r in range(rows) if grid[r][c] == 1)
        if ones_count >= rows * 0.9:  # 90% or more are 1s
            sep_col = c
            break

    if sep_col is not None:
        # Vertical separator - match by row
        for r in range(rows):
            # Find template colors in this row (excluding 0, 1, 5)
            row_templates = []
            for c in range(cols):
                val = grid[r][c]
                if val != 0 and val != 1 and val != 5:
                    row_templates.append((c, val))

            if not row_templates:
                # No template in this row, find nearest row with template
                nearest_color = None
                min_row_dist = float('inf')
                for rr in range(rows):
                    for c in range(cols):
                        val = grid[rr][c]
                        if val != 0 and val != 1 and val != 5:
                            if abs(rr - r) < min_row_dist:
                                min_row_dist = abs(rr - r)
                                nearest_color = val
                if nearest_color:
                    row_templates = [(0, nearest_color)]

            # Use the template with lowest column index
            if row_templates:
                row_templates.sort()
                template_color = row_templates[0][1]

                # Color all 5s in this row
                for c in range(cols):
                    if grid[r][c] == 5:
                        result[r][c] = template_color

    elif sep_row is not None:
        # Horizontal separator - match by column
        for c in range(cols):
            # Find template colors in this column
            col_templates = []
            for r in range(rows):
                val = grid[r][c]
                if val != 0 and val != 1 and val != 5:
                    col_templates.append((r, val))

            if not col_templates:
                # No template in this column, find nearest column with template
                nearest_color = None
                min_col_dist = float('inf')
                for cc in range(cols):
                    for r in range(rows):
                        val = grid[r][cc]
                        if val != 0 and val != 1 and val != 5:
                            if abs(cc - c) < min_col_dist:
                                min_col_dist = abs(cc - c)
                                nearest_color = val
                if nearest_color:
                    col_templates = [(0, nearest_color)]

            # Use the template with lowest row index
            if col_templates:
                col_templates.sort()
                template_color = col_templates[0][1]

                # Color all 5s in this column
                for r in range(rows):
                    if grid[r][c] == 5:
                        result[r][c] = template_color

    return result
