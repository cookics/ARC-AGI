def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Frames: 2-wide bars (horizontal or vertical) with values 1,2,3,4
    2. Markers: isolated values != 8 and not in frames
    3. Base side: the side with value 2 (indicates projection origin)
    4. Horizontal frames project 3 vertical beams downward
    5. Vertical frames project 3 horizontal beams leftward/rightward
    6. Beam selection based on opposite side values (3 or 4)
    7. Marker assignment: ascending for vertical beams, reversed for horizontal beams

    Procedure:
    1. Find all frames and determine base side
    2. Collect and sort markers
    3. For each frame, select 3 projection lines based on structure
    4. Fill beams with appropriate marker values
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find frames
    frames = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) in visited or grid[r][c] == 8:
                continue

            # Try horizontal frame (2 rows)
            if r + 1 < rows and grid[r][c] != 8 and grid[r + 1][c] != 8:
                c_end = c
                while c_end + 1 < cols and grid[r][c_end + 1] != 8 and grid[r + 1][c_end + 1] != 8:
                    c_end += 1

                if c_end - c >= 2:
                    for rr in [r, r + 1]:
                        for cc in range(c, c_end + 1):
                            visited.add((rr, cc))
                    frames.append(('horizontal', r, r + 1, c, c_end))
                    continue

            # Try vertical frame (2 cols)
            if c + 1 < cols and (r, c) not in visited and grid[r][c + 1] != 8:
                r_end = r
                while r_end + 1 < rows and grid[r_end + 1][c] != 8 and grid[r_end + 1][c + 1] != 8:
                    r_end += 1

                if r_end - r >= 2:
                    for rr in range(r, r_end + 1):
                        for cc in [c, c + 1]:
                            visited.add((rr, cc))
                    frames.append(('vertical', r, r_end, c, c + 1))

    # Find markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] != 8:
                markers.append(grid[r][c])

    markers.sort()

    # Process each frame
    for frame_info in frames:
        if frame_info[0] == 'horizontal':
            _, r1, r2, c1, c2 = frame_info

            # Determine base row (has more 2s)
            count_2_r1 = sum(1 for c in range(c1, c2 + 1) if grid[r1][c] == 2)
            count_2_r2 = sum(1 for c in range(c1, c2 + 1) if grid[r2][c] == 2)

            if count_2_r2 > count_2_r1:
                base_row, other_row = r2, r1
            else:
                base_row, other_row = r1, r2

            # Find projection columns based on opposite side values
            cols_with_4 = []
            cols_with_3 = []

            for c in range(c1, c2 + 1):
                if grid[base_row][c] == 2:
                    other_val = grid[other_row][c]
                    if other_val == 4:
                        cols_with_4.append(c)
                    elif other_val == 3 or other_val == 1:
                        cols_with_3.append(c)

            # Select 3 columns: last col with 4, first 2 cols with 3
            proj_cols = []
            if cols_with_4:
                proj_cols.append(cols_with_4[-1])  # Last col with 4
            if len(cols_with_3) >= 2:
                proj_cols[0:0] = cols_with_3[:2]  # First 2 cols with 3 (prepend)
            elif cols_with_3:
                proj_cols.insert(0, cols_with_3[0])

            # Project downward with ascending markers
            for i in range(min(3, len(proj_cols))):
                if i < len(markers):
                    col = proj_cols[i]
                    val = markers[i]
                    for r in range(base_row + 1, rows):
                        if result[r][col] == 8:
                            result[r][col] = val
                        else:
                            break

        else:  # vertical
            _, r1, r2, c1, c2 = frame_info

            # Determine base col (has more 2s)
            count_2_c1 = sum(1 for r in range(r1, r2 + 1) if grid[r][c1] == 2)
            count_2_c2 = sum(1 for r in range(r1, r2 + 1) if grid[r][c2] == 2)

            if count_2_c1 > count_2_c2:
                base_col, other_col = c1, c2
                base_is_left = True
            else:
                base_col, other_col = c2, c1
                base_is_left = False

            # Project direction: base on left projects left, base on right projects left too (for now)
            # But we'll use different marker ordering

            # Find projection rows - check ALL rows in frame, categorize by base_col value
            rows_with_base_4 = []
            rows_with_base_2 = []

            for r in range(r1, r2 + 1):
                base_val = grid[r][base_col]
                if base_val == 4:
                    rows_with_base_4.append(r)
                elif base_val == 2:
                    rows_with_base_2.append(r)

            # Select 3 rows: 1 from rows with base=4, 2 from rows with base=2
            proj_rows = []

            # Select one row with base_val=4 (avoid top edge)
            if rows_with_base_4:
                for row in rows_with_base_4:
                    if row != r1:
                        proj_rows.append(row)
                        break
                else:
                    if len(rows_with_base_4) > 1:
                        proj_rows.append(rows_with_base_4[1])

            # Select 2 rows from rows with base_val=2
            if len(rows_with_base_2) >= 2:
                # Take every other row starting from index 1
                if len(rows_with_base_2) >= 4:
                    proj_rows.extend([rows_with_base_2[1], rows_with_base_2[3]])
                else:
                    # For shorter lists, take last 2
                    proj_rows.extend(rows_with_base_2[-2:])
            elif rows_with_base_2:
                proj_rows.append(rows_with_base_2[0])

            # Determine projection direction based on frame position in grid
            frame_midpoint = (c1 + c2) / 2
            grid_midpoint = cols / 2
            project_left = (frame_midpoint < grid_midpoint)

            # Marker ordering: depends on base position AND frame position
            if project_left and not base_is_left:
                # Left side frame with base on right: ascending
                use_markers = markers
            else:
                # All other cases: reversed
                use_markers = list(reversed(markers))

            # Project in the determined direction
            for i in range(min(3, len(proj_rows))):
                if i < len(use_markers):
                    row = proj_rows[i]
                    val = use_markers[i]
                    if project_left:
                        # Project left
                        start_col = min(c1, c2) - 1
                        for c in range(start_col, -1, -1):
                            if result[row][c] == 8:
                                result[row][c] = val
                            else:
                                break
                    else:
                        # Project right
                        start_col = max(c1, c2) + 1
                        for c in range(start_col, cols):
                            if result[row][c] == 8:
                                result[row][c] = val
                            else:
                                break

    return result
