def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 8s (sources) and 2s (section markers)
    2. 2s can be horizontal (same row) or vertical (same column)
    3. 2s define section boundaries for propagation
    4. 8s propagate perpendicular to 2s with shifts at section boundaries

    Procedure:
    1. Find all 8s and 2s
    2. Determine orientation (vertical 2s vs horizontal 2s)
    3. Create sections based on 2 positions
    4. For each 8, propagate with appropriate shifts
       - Horizontal 2s: 8s propagate upward, shifting through column sections
       - Vertical 2s: 8s propagate downward, shifting by 1 column per section
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find 8s and 2s
    eights = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 8]
    twos = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 2]

    if not eights or not twos:
        return result

    # Determine orientation
    two_rows = [r for r, c in twos]
    two_cols = [c for r, c in twos]

    if len(set(two_rows)) == 1:
        # Horizontal 2s (same row) - propagate upward through column sections
        two_cols_sorted = sorted(set(two_cols))

        # Create column sections
        sections = []
        start = 0
        for col in two_cols_sorted:
            sections.append((start, col - 1))
            start = col
        sections.append((start, cols - 1))

        # Determine shift direction based on 8s position
        eight_col = eights[0][1]  # All 8s should be in the same column
        if eight_col < cols / 2:
            shift_dir = 1  # Left side, shift right
        else:
            shift_dir = -1  # Right side, shift left

        # For each 8, propagate upward shifting through sections
        for r8, c8 in eights:
            # Find which section the 8 is in
            section_idx = -1
            for i, (s_start, s_end) in enumerate(sections):
                if s_start <= c8 <= s_end:
                    section_idx = i
                    break

            if section_idx == -1:
                continue

            # Propagate upward, shifting by shift_dir per row
            current_row = r8
            current_section = section_idx

            while current_row >= 0 and 0 <= current_section < len(sections):
                s_start, s_end = sections[current_section]
                for c in range(s_start, s_end + 1):
                    if result[current_row][c] != 2:
                        result[current_row][c] = 8
                current_row -= 1
                current_section += shift_dir

    elif len(set(two_cols)) == 1:
        # Vertical 2s (same column) - propagate downward shifting by 1 column
        two_col = two_cols[0]
        two_rows_sorted = sorted(set(two_rows))

        # Determine shift direction based on 2s position
        if two_col == 0:
            shift_dir = 1  # Left edge, shift right
        else:
            shift_dir = -1  # Right edge, shift left

        # Create row sections
        sections = []
        start = 0
        for row in two_rows_sorted:
            sections.append((start, row - 1))
            start = row
        sections.append((start, rows - 1))

        # For each 8, propagate downward shifting by 1 column per section
        for r8, c8 in eights:
            # Find which section the 8 is in
            section_idx = -1
            for i, (s_start, s_end) in enumerate(sections):
                if s_start <= r8 <= s_end:
                    section_idx = i
                    break

            if section_idx == -1:
                continue

            # Propagate downward through all sections
            current_col = c8
            for i in range(section_idx, len(sections)):
                s_start, s_end = sections[i]
                # Fill this section (only from r8 onwards in the first section)
                start_row = r8 if i == section_idx else s_start
                for r in range(start_row, s_end + 1):
                    if 0 <= current_col < cols and result[r][current_col] != 2:
                        result[r][current_col] = 8
                # Shift for next section
                current_col += shift_dir

    return result
