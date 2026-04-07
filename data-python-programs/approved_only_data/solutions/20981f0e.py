def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input grid contains 2s that form a regular grid pattern, dividing the space into rectangular sections.
    2. Within each section, there are 1s that need to be rearranged.
    3. The output shows that within each section, all 1s are moved so that their centroid aligns with the center of that section.

    Procedure:
    1. Find all positions of 2s to determine section boundaries
    2. For each section, collect all 1s within that section
    3. Calculate the centroid of the 1s and the center of the section
    4. Calculate the offset needed to move the centroid to the section center
    5. Apply this offset (rounded to integers) to all 1s in the section
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the input grid

    # Find all 2s to determine section boundaries
    twos_rows = set()
    twos_cols = set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos_rows.add(r)
                twos_cols.add(c)

    # Sort to get boundaries in order
    row_boundaries = sorted(twos_rows)
    col_boundaries = sorted(twos_cols)

    # Add grid boundaries if not already present
    if 0 not in row_boundaries and any(grid[0][c] != 2 for c in range(cols)):
        row_boundaries = [-1] + row_boundaries
    if rows - 1 not in row_boundaries and any(
        grid[rows - 1][c] != 2 for c in range(cols)
    ):
        row_boundaries = row_boundaries + [rows]
    if 0 not in col_boundaries and any(grid[r][0] != 2 for r in range(rows)):
        col_boundaries = [-1] + col_boundaries
    if cols - 1 not in col_boundaries and any(
        grid[r][cols - 1] != 2 for r in range(rows)
    ):
        col_boundaries = col_boundaries + [cols]

    # Process each section
    for i in range(len(row_boundaries) - 1):
        for j in range(len(col_boundaries) - 1):
            # Define section boundaries
            r1, r2 = row_boundaries[i], row_boundaries[i + 1]
            c1, c2 = col_boundaries[j], col_boundaries[j + 1]

            # Adjust boundaries to exclude the boundary lines
            section_r1 = max(0, r1 + 1)
            section_r2 = min(rows, r2)
            section_c1 = max(0, c1 + 1)
            section_c2 = min(cols, c2)

            # Find all 1s in this section
            ones_in_section = []
            for r in range(section_r1, section_r2):
                for c in range(section_c1, section_c2):
                    if grid[r][c] == 1:
                        ones_in_section.append((r, c))

            if not ones_in_section:
                continue

            # Calculate centroid of 1s
            centroid_r = sum(r for r, c in ones_in_section) / len(ones_in_section)
            centroid_c = sum(c for r, c in ones_in_section) / len(ones_in_section)

            # Calculate center of section
            section_center_r = (section_r1 + section_r2 - 1) / 2
            section_center_c = (section_c1 + section_c2 - 1) / 2

            # Calculate offset
            offset_r = section_center_r - centroid_r
            offset_c = section_center_c - centroid_c

            # Round offset to integers
            offset_r = round(offset_r)
            offset_c = round(offset_c)

            # Clear 1s from this section in result
            for r in range(section_r1, section_r2):
                for c in range(section_c1, section_c2):
                    if result[r][c] == 1:
                        result[r][c] = 0

            # Place 1s at new positions
            for r, c in ones_in_section:
                new_r = r + offset_r
                new_c = c + offset_c
                if (
                    section_r1 <= new_r < section_r2
                    and section_c1 <= new_c < section_c2
                ):
                    result[new_r][new_c] = 1

    return result
