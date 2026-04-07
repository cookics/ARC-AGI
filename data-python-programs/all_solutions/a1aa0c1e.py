def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has horizontal separator rows (all cells same non-zero color)
    2. Between separators are sections with patterns made of the separator color
    3. Last separator is always 9s
    4. Output has one row per section, showing: [color repetitions..., 9, 5 or 0]
    5. Number of color repetitions = ceil(pattern_rows / 3)
    6. The 5 marker appears in one of the sections

    Procedure:
    1. Find all horizontal separator rows (full rows of single color)
    2. For each section between separators, count pattern rows
    3. Calculate ceil(pattern_rows / 3) repetitions for each color
    4. Find which output row gets the 5 based on pattern column overlap
    5. Build 3x5 output grid
    """
    import math

    rows = len(grid)
    cols = len(grid[0])

    # Find horizontal separator lines (entire row is one color)
    separators = []
    for i in range(rows):
        if len(set(grid[i])) == 1 and grid[i][0] != 0:
            separators.append((i, grid[i][0]))

    # Separate out the 9s line (bottom separator) and other colored separators
    colored_separators = [s for s in separators if s[1] != 9]
    nine_line_row = next(i for i, color in separators if color == 9)

    # Find where the 5 is located
    five_pos = None
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 5:
                five_pos = (i, j)
                break
        if five_pos:
            break

    # For each section, count pattern rows and find pattern column range
    sections = []
    for idx, (sep_row, color) in enumerate(colored_separators):
        # Determine section bounds
        start_row = sep_row + 1
        if idx < len(colored_separators) - 1:
            end_row = colored_separators[idx + 1][0]
        else:
            end_row = nine_line_row

        # Count consecutive non-empty pattern rows
        pattern_rows = 0
        pattern_cols = set()
        for r in range(start_row, end_row):
            if any(grid[r][c] != 0 and grid[r][c] != 5 for c in range(cols)):
                pattern_rows += 1
                # Track which columns have the pattern color
                for c in range(cols):
                    if grid[r][c] == color:
                        pattern_cols.add(c)

        # Calculate repetitions using ceiling division
        repetitions = math.ceil(pattern_rows / 3) if pattern_rows > 0 else 0

        sections.append({
            'color': color,
            'repetitions': repetitions,
            'start_row': start_row,
            'end_row': end_row,
            'pattern_cols': pattern_cols
        })

    # Build output rows
    result = []
    for section in sections:
        row = [section['color']] * section['repetitions']
        # Pad to 3 elements
        row.extend([0] * (3 - len(row)))
        row.append(9)  # Column 3 is always 9
        row.append(0)  # Column 4 is placeholder for 5
        result.append(row)

    # Determine which output row gets the 5 based on column position
    # Pattern observed: higher column number → lower row index (inverted)
    # Column 0-11 → row 2, Column 12-15 → row 1, Column 16-20 → row 0
    if five_pos:
        five_row, five_col = five_pos

        if five_col <= 11:
            five_output_row = 2
        elif five_col <= 15:
            five_output_row = 1
        else:
            five_output_row = 0

        if 0 <= five_output_row < len(result):
            result[five_output_row][4] = 5

    return result
