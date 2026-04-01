def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with rectangular sections separated by rows/columns of 0s
    2. Output is a condensed grid showing transformed rectangular sections
    3. Each rectangle has a repeating pattern with a dominant background color
    4. Some rows contain special colors (non-background, non-zero values) as markers
    5. Special colors indicate how rectangles should be colored in the output

    Procedure:
    1. Find rectangular sections by identifying rows/columns that are all 0s
    2. Extract template pattern from first rectangle and identify background color
    3. For each row, collect special colors in left-to-right order
    4. Assign collected colors to output rectangles in order
    5. Generate output by inverting the template pattern with assigned colors
    """

    rows, cols = len(grid), len(grid[0])

    # Find section boundaries (rows/cols that are all 0s)
    zero_rows = []
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            zero_rows.append(r)

    zero_cols = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            zero_cols.append(c)

    # Determine section boundaries
    section_row_bounds = []
    prev = 0
    for r in zero_rows:
        if r > prev:
            section_row_bounds.append((prev, r - 1))
        prev = r + 1
    if prev < rows:
        section_row_bounds.append((prev, rows - 1))

    section_col_bounds = []
    prev = 0
    for c in zero_cols:
        if c > prev:
            section_col_bounds.append((prev, c - 1))
        prev = c + 1
    if prev < cols:
        section_col_bounds.append((prev, cols - 1))

    if not section_row_bounds or not section_col_bounds:
        return [[]]

    # Find background color (most common non-zero)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != 0:
                color_counts[val] = color_counts.get(val, 0) + 1

    if not color_counts:
        return [[]]

    background = max(color_counts.keys(), key=lambda x: color_counts[x])

    # Extract template pattern from first rectangle
    first_rect_r = section_row_bounds[0]
    first_rect_c = section_col_bounds[0]
    template = []
    for r in range(first_rect_r[0], first_rect_r[1] + 1):
        template_row = []
        for c in range(first_rect_c[0], first_rect_c[1] + 1):
            template_row.append(grid[r][c])
        template.append(template_row)

    # Find rows containing special colors and extract patterns
    special_row_patterns = []

    for row_bounds in section_row_bounds:
        for r in range(row_bounds[0], row_bounds[1] + 1):
            # Collect special colors in left-to-right order by position
            special_colors_ordered = []
            for c in range(cols):
                if grid[r][c] != 0 and grid[r][c] != background:
                    special_colors_ordered.append(grid[r][c])

            if special_colors_ordered:
                # Assign colors to rectangles in order (pad with background if needed)
                row_pattern = []
                for i in range(len(section_col_bounds)):
                    if i < len(special_colors_ordered):
                        row_pattern.append(special_colors_ordered[i])
                    else:
                        row_pattern.append(background)

                special_row_patterns.append(row_pattern)

    if not special_row_patterns:
        return [[]]

    # Create output
    result = []
    template_height = len(template)
    template_width = len(template[0])

    for i, pattern in enumerate(special_row_patterns):
        if i > 0:
            # Add separator row
            separator_width = template_width * len(pattern) + (
                len(pattern) - 1
            )  # Include separator columns
            separator = [background] * separator_width
            result.append(separator)

        # Create transformed section
        for tr in range(template_height):
            output_row = []
            for rect_idx in range(len(pattern)):
                rect_color = pattern[rect_idx]
                for tc in range(template_width):
                    if template[tr][tc] == background:
                        output_row.append(rect_color)
                    else:
                        output_row.append(background)

                # Add separator column between rectangles (except after the last one)
                if rect_idx < len(pattern) - 1:
                    output_row.append(background)

            result.append(output_row)

    return result
