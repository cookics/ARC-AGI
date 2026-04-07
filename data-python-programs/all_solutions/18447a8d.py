def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The grid has sections separated by rows of all 7s. Each section has 8s on the left
    and some other color on the right. The colors rotate between sections based on rules:
    - 2 sections: no rotation
    - 3 sections: cycle all colors (A→B→C→A)
    - 4 sections: first stays same, others cycle backward (A→A, B→D, C→B, D→C)

    In output, 8s stay in place, new colors fill positions right after 8s with same counts.

    Procedure:
    1. Find section boundaries (rows of all 7s)
    2. Extract unique colors and determine rotation mapping
    3. For each section, replace old color with new color right after 8s
    """

    # Step 1: Find section boundaries
    section_boundaries = []
    for i, row in enumerate(grid):
        if all(cell == 7 for cell in row):
            section_boundaries.append(i)

    # Extract sections (start_row, end_row) pairs
    sections = []
    start = 0
    for boundary in section_boundaries:
        if start < boundary:
            sections.append((start, boundary - 1))
        start = boundary + 1
    if start < len(grid):
        sections.append((start, len(grid) - 1))

    # Step 2: Find unique colors (excluding 7 and 8)
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 7 and cell != 8:
                colors.add(cell)

    sorted_colors = sorted(colors)

    # Create color mapping based on number of colors
    color_mapping = {}
    if len(sorted_colors) == 2:
        # No rotation
        for color in sorted_colors:
            color_mapping[color] = color
    elif len(sorted_colors) == 3:
        # Forward cycle: A→B→C→A
        for i in range(3):
            color_mapping[sorted_colors[i]] = sorted_colors[(i + 1) % 3]
    elif len(sorted_colors) == 4:
        # First stays, others cycle backward: A→A, B→D, C→B, D→C
        color_mapping[sorted_colors[0]] = sorted_colors[0]
        color_mapping[sorted_colors[1]] = sorted_colors[3]
        color_mapping[sorted_colors[2]] = sorted_colors[1]
        color_mapping[sorted_colors[3]] = sorted_colors[2]

    # Step 3: Find original color for each section
    section_colors = []
    for start_row, end_row in sections:
        section_color = None
        for row_idx in range(start_row, end_row + 1):
            for cell in grid[row_idx]:
                if cell != 7 and cell != 8:
                    section_color = cell
                    break
            if section_color:
                break
        section_colors.append(section_color)

    # Step 4: Count original color cells per row for each color
    color_row_counts = {}
    for i, (start_row, end_row) in enumerate(sections):
        color = section_colors[i]
        color_row_counts[color] = []
        for row_idx in range(start_row, end_row + 1):
            count = sum(1 for cell in grid[row_idx] if cell == color)
            color_row_counts[color].append(count)

    # Step 5: Create output grid
    result = [row[:] for row in grid]

    for i, (start_row, end_row) in enumerate(sections):
        old_color = section_colors[i]
        new_color = color_mapping[old_color]
        new_color_counts = color_row_counts[new_color]

        for j, row_idx in enumerate(range(start_row, end_row + 1)):
            # Clear old color
            for col_idx in range(len(result[row_idx])):
                if result[row_idx][col_idx] == old_color:
                    result[row_idx][col_idx] = 7

            # Find rightmost 8
            rightmost_8 = -1
            for col_idx in range(len(result[row_idx])):
                if result[row_idx][col_idx] == 8:
                    rightmost_8 = col_idx

            # Place new color cells after rightmost 8
            count_to_place = new_color_counts[j]
            for k in range(count_to_place):
                if rightmost_8 + 1 + k < len(result[row_idx]):
                    result[row_idx][rightmost_8 + 1 + k] = new_color

    return result
