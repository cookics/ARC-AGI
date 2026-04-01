def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with certain values forming complete horizontal and vertical separator lines
    2. These separator lines divide the grid into rectangular sections
    3. Each section contains predominantly one non-separator value
    4. Output is a smaller grid where each cell represents the dominant value of a section
    5. Output dimensions equal the number of sections created horizontally and vertically

    Procedure:
    1. Identify the separator value by finding which value forms the most complete rows and columns
    2. Find all horizontal separator rows and vertical separator columns
    3. Use separator positions to determine section boundaries
    4. For each section, extract the most frequent non-separator value
    5. Construct output grid with dominant values from each section
    """

    rows, cols = len(grid), len(grid[0])

    # Count frequency of each value to identify separator
    from collections import Counter

    all_values = []
    for r in range(rows):
        for c in range(cols):
            all_values.append(grid[r][c])

    value_counts = Counter(all_values)

    # The separator is likely the value that forms complete lines
    # Let's check which value forms the most complete rows/columns
    separator_value = None
    max_lines = 0

    for value, count in value_counts.items():
        # Check how many complete rows/columns this value forms
        complete_rows = 0
        complete_cols = 0

        # Check rows
        for r in range(rows):
            if all(grid[r][c] == value for c in range(cols)):
                complete_rows += 1

        # Check columns
        for c in range(cols):
            if all(grid[r][c] == value for r in range(rows)):
                complete_cols += 1

        total_lines = complete_rows + complete_cols
        if total_lines > max_lines:
            max_lines = total_lines
            separator_value = value

    if separator_value is None:
        # Fallback: use most frequent value
        separator_value = value_counts.most_common(1)[0][0]

    # Find separator rows and columns
    separator_rows = []
    separator_cols = []

    for r in range(rows):
        if all(grid[r][c] == separator_value for c in range(cols)):
            separator_rows.append(r)

    for c in range(cols):
        if all(grid[r][c] == separator_value for r in range(rows)):
            separator_cols.append(c)

    # Create section boundaries
    row_boundaries = [0] + separator_rows + [rows]
    col_boundaries = [0] + separator_cols + [cols]

    row_boundaries = sorted(list(set(row_boundaries)))
    col_boundaries = sorted(list(set(col_boundaries)))

    # Extract sections
    result = []

    for i in range(len(row_boundaries) - 1):
        result_row = []
        start_r = row_boundaries[i]
        end_r = row_boundaries[i + 1]

        # Skip if this is a separator row
        if start_r in separator_rows:
            start_r += 1
        if end_r - 1 in separator_rows:
            end_r -= 1

        if start_r >= end_r:
            continue

        for j in range(len(col_boundaries) - 1):
            start_c = col_boundaries[j]
            end_c = col_boundaries[j + 1]

            # Skip if this is a separator column
            if start_c in separator_cols:
                start_c += 1
            if end_c - 1 in separator_cols:
                end_c -= 1

            if start_c >= end_c:
                continue

            # Extract the dominant value from this section
            section_values = []
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    if grid[r][c] != separator_value:
                        section_values.append(grid[r][c])

            if section_values:
                # Get most common non-separator value
                section_counter = Counter(section_values)
                dominant_value = section_counter.most_common(1)[0][0]
                result_row.append(dominant_value)

        if result_row:
            result.append(result_row)

    return result
