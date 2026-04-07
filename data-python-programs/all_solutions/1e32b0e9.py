def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 17x17 grid divided by lines of a specific color into a 3x3 arrangement of 5x5 sections.
    2. Output preserves the same grid structure with divider lines intact.
    3. One section contains a complete pattern while others have partial or no patterns.
    4. The complete pattern gets replicated to all other sections using the divider color.
    5. Existing pattern elements in other sections are preserved in their original color.

    Procedure:
    1. Identify the divider color that creates the 3x3 grid structure.
    2. Extract all 9 sections (each 5x5) from the grid.
    3. Find the section with the most complete pattern by counting non-zero, non-divider elements.
    4. Use the most complete section as a template pattern.
    5. Apply the template to all sections, filling empty spaces with divider color while preserving existing pattern elements.
    """

    result = [row[:] for row in grid]  # deep copy

    # Find divider color (appears in rows 5, 11 and cols 5, 11)
    divider_color = grid[5][0]

    # Extract the 9 sections (each 5x5)
    sections = []
    for section_row in range(3):
        section_row_list = []
        for section_col in range(3):
            start_row = section_row * 6
            start_col = section_col * 6
            section = []
            for r in range(5):
                row = []
                for c in range(5):
                    row.append(grid[start_row + r][start_col + c])
                section.append(row)
            section_row_list.append(section)
        sections.append(section_row_list)

    # Find the section with the most complete pattern
    max_pattern_count = 0
    template_section_row, template_section_col = 0, 0

    for section_row in range(3):
        for section_col in range(3):
            section = sections[section_row][section_col]
            pattern_count = 0
            for r in range(5):
                for c in range(5):
                    if section[r][c] != 0 and section[r][c] != divider_color:
                        pattern_count += 1

            if pattern_count > max_pattern_count:
                max_pattern_count = pattern_count
                template_section_row, template_section_col = section_row, section_col

    # Get the template pattern
    template = sections[template_section_row][template_section_col]

    # Apply the template to all sections
    for section_row in range(3):
        for section_col in range(3):
            start_row = section_row * 6
            start_col = section_col * 6

            for r in range(5):
                for c in range(5):
                    template_value = template[r][c]
                    current_value = result[start_row + r][start_col + c]

                    if template_value != 0 and template_value != divider_color:
                        # There's a pattern element in the template
                        if current_value == 0:
                            # Empty space - fill with divider color
                            result[start_row + r][start_col + c] = divider_color
                        # else: preserve existing pattern element

    return result
