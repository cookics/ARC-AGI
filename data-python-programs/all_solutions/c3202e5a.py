def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid partitioned by separator rows and columns (all same non-zero value)
    2. Output is one of the sections with separator values replaced by 0
    3. The selected section contains the most occurrences of the target value
    4. Target value is the most frequent non-zero, non-separator value across all sections

    Procedure:
    1. Find separator value and extract sections
    2. Find target value (most frequent non-separator value)
    3. Search for a section that matches the expected pattern
    4. Return that section with separators replaced by 0
    """

    rows, cols = len(grid), len(grid[0])

    # Find separator value and positions
    separator = None
    separator_rows = []
    separator_cols = []

    # Find separator rows
    for r in range(rows):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            separator = grid[r][0]
            separator_rows.append(r)

    # Find separator columns
    for c in range(cols):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1 and col_vals[0] != 0:
            separator_cols.append(c)

    # Create section boundaries
    row_sections = []
    start = 0
    for sep_row in separator_rows:
        if start < sep_row:
            row_sections.append((start, sep_row - 1))
        start = sep_row + 1
    if start < rows:
        row_sections.append((start, rows - 1))

    col_sections = []
    start = 0
    for sep_col in separator_cols:
        if start < sep_col:
            col_sections.append((start, sep_col - 1))
        start = sep_col + 1
    if start < cols:
        col_sections.append((start, cols - 1))

    # Extract all sections
    sections = []
    for r_start, r_end in row_sections:
        section_row = []
        for c_start, c_end in col_sections:
            section = []
            for r in range(r_start, r_end + 1):
                row = []
                for c in range(c_start, c_end + 1):
                    row.append(grid[r][c])
                section.append(row)
            section_row.append(section)
        sections.append(section_row)

    # Get section dimensions
    section_height = len(sections[0][0])
    section_width = len(sections[0][0][0])
    num_section_rows = len(sections)
    num_section_cols = len(sections[0])

    # Count frequency of each non-zero, non-separator value
    value_counts = {}
    for section_row in sections:
        for section in section_row:
            for row in section:
                for val in row:
                    if val != 0 and val != separator:
                        value_counts[val] = value_counts.get(val, 0) + 1

    # Find target value (most frequent)
    if not value_counts:
        result = [[0]]
        return result
    target = max(value_counts.keys(), key=lambda x: value_counts[x])

    # Search through all sections to find one that contains the target
    # in the right pattern for the expected output
    best_section = None
    max_target_count = 0

    for sr in range(num_section_rows):
        for sc in range(num_section_cols):
            section = sections[sr][sc]
            # Count target values in this section
            target_count = 0
            for row in section:
                for val in row:
                    if val == target:
                        target_count += 1

            # Keep track of section with most target values
            if target_count > max_target_count:
                max_target_count = target_count
                best_section = section

    # If we found a good section, use it; otherwise use first section
    if best_section is None:
        best_section = sections[0][0]

    # Return section with separators replaced by 0
    result = []
    for row in best_section:
        new_row = []
        for val in row:
            if val == separator:
                new_row.append(0)
            else:
                new_row.append(val)
        result.append(new_row)

    return result
