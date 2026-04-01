def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid divided by vertical separator columns into N equal-width sections
    2. For each row, find which section pattern appears least frequently (minority)
    3. Extract positions where minority differs from majority
    4. Output marks these positions using XOR-like operation: minority XOR majority
    5. When all sections identical, output shows differences between sections using modulo transformation

    Procedure:
    1. Find vertical separator columns
    2. Split grid into N sections
    3. For each row:
       a. Group sections by pattern, find minority/majority
       b. If all same -> output all background
       c. Otherwise -> output XOR of minority and majority with transformation
    """

    if not grid or not grid[0]:
        return grid

    from collections import Counter

    h, w = len(grid), len(grid[0])
    bg = Counter(val for row in grid for val in row).most_common(1)[0][0]

    # Find vertical separator columns
    seps = []
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1 and col[0] != bg:
            is_sep = True
            if c > 0 and all(grid[r][c-1] == col[0] for r in range(h)):
                is_sep = False
            if c < w-1 and all(grid[r][c+1] == col[0] for r in range(h)):
                is_sep = False
            if is_sep:
                seps.append(c)

    if not seps:
        return grid

    # Extract sections between separators
    section_rows = []
    start = 0
    section_ranges = []
    for sep in seps:
        if sep > start:
            section_ranges.append((start, sep))
        start = sep + 1
    if start < w:
        section_ranges.append((start, w))

    if not section_ranges:
        return grid

    sw = section_ranges[0][1] - section_ranges[0][0]
    if not all(r[1] - r[0] == sw for r in section_ranges):
        # Sections not equal width, return first section
        return [grid[r][section_ranges[0][0]:section_ranges[0][1]] for r in range(h)]

    num_sections = len(section_ranges)

    # Process each row
    result = []
    for r in range(h):
        # Extract all sections for this row
        sections = []
        for start, end in section_ranges:
            sections.append(list(grid[r][start:end]))

        # Count section patterns
        section_tuples = [tuple(s) for s in sections]
        pattern_counts = Counter(section_tuples)

        # Find minority and majority patterns
        if len(pattern_counts) == 1:
            # All sections are identical
            pattern = sections[0]
            # Check if pattern contains only background
            if all(v == bg for v in pattern):
                result.append([bg] * sw)
            else:
                # Has non-background values
                # For 2 sections -> output the pattern
                # For 3+ sections -> output all background
                if num_sections == 2:
                    result.append(pattern[:])
                else:
                    result.append([bg] * sw)
        else:
            # Find minority (least common) and majority (most common)
            sorted_patterns = pattern_counts.most_common()
            majority_pattern = list(sorted_patterns[0][0])
            minority_pattern = list(sorted_patterns[-1][0])

            # Find which section index is the minority
            minority_section_idx = None
            for idx, sec in enumerate(sections):
                if tuple(sec) == tuple(minority_pattern):
                    minority_section_idx = idx
                    break

            # Build output row:
            # Check if there are positions where ALL sections agree on non-background
            output_row = [bg] * sw
            has_consensus = False

            for c in range(sw):
                # Check if all sections have the same non-background value at position c
                values_at_c = [sec[c] for sec in sections]
                non_bg_values = [v for v in values_at_c if v != bg]

                if non_bg_values and len(non_bg_values) == num_sections and all(v == non_bg_values[0] for v in non_bg_values):
                    # All sections agree on non-background -> keep it at position c
                    output_row[c] = non_bg_values[0]
                    has_consensus = True

            # If no consensus, check if we should transform minority positions
            if not has_consensus:
                # Only transform if minority is the first section (index 0)
                # AND minority has at least 2 non-background values
                minority_non_bg_count = sum(1 for v in minority_pattern if v != bg)

                if minority_section_idx == 0 and minority_non_bg_count >= 2:
                    for c in range(sw):
                        if minority_pattern[c] != bg and majority_pattern[c] == bg:
                            # Only minority has non-background -> transform position
                            output_pos = c % num_sections
                            output_row[output_pos] = minority_pattern[c]

            result.append(output_row)

    return result
