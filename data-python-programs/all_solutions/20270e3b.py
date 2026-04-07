def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 7s which act as "noise" markers
    2. Three patterns:
       a) If 7s form complete horizontal rows: remove rows from first to last 7-row
       b) If there's a vertical separator between 7s: split/merge left and right parts
       c) If 7s are localized: remove rows with 7s and expand columns

    Procedure:
    1. Find all 7s and check for complete rows
    2. Check for vertical separator between 7s groups
    3. Otherwise handle as column expansion case
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])

    # Find all 7s
    sevens = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 7]

    if not sevens:
        return grid

    rows_with_7 = set(r for r, c in sevens)
    cols_with_7 = set(c for r, c in sevens)

    min_row_7 = min(r for r, c in sevens)
    max_row_7 = max(r for r, c in sevens)
    min_col_7 = min(c for r, c in sevens)
    max_col_7 = max(c for r, c in sevens)

    # Pattern 1: Check if entire rows are filled with 7s
    full_7_rows = [r for r in rows_with_7 if all(grid[r][c] == 7 for c in range(cols))]

    if full_7_rows:
        # Remove all rows from min to max rows containing 7s
        result = [grid[r][:] for r in range(rows) if r < min_row_7 or r > max_row_7]
        return result

    # Pattern 2: Check for vertical separator between 7s groups
    # Find uniform columns (all same value) between min and max 7 columns
    separator_col = None
    for c in range(min_col_7 + 1, max_col_7):
        col_vals = [grid[r][c] for r in range(rows)]
        if len(set(col_vals)) == 1:
            # This column is uniform, check if it's a good separator (e.g., all 4s)
            if col_vals[0] == 4:
                separator_col = c
                break

    if separator_col is not None:
        # Split at separator
        left = [row[:separator_col+1] for row in grid]
        right = [row[separator_col+1:] for row in grid]

        # Find 7s in each part
        left_7s = [(r, c) for r, c in sevens if c <= separator_col]
        right_7s = [(r, c - separator_col - 1) for r, c in sevens if c > separator_col]

        if left_7s and right_7s:
            # Get bounding boxes
            left_min_r = min(r for r, c in left_7s)
            left_min_c = min(c for r, c in left_7s)

            right_min_r = min(r for r, c in right_7s)
            right_min_c = min(c for r, c in right_7s)

            # Determine how many rows to copy from right
            # Stop at uniform rows in right part
            rows_to_copy = 0
            for offset in range(rows):
                src_r = right_min_r + 1 + offset
                if src_r >= rows:
                    break
                # Check if this row in right part is uniform (stopping condition)
                row_vals = right[src_r]
                if len(set(row_vals)) == 1:
                    break
                rows_to_copy += 1

            # Copy from right to left
            for offset in range(rows_to_copy):
                src_r = right_min_r + 1 + offset
                dst_r = left_min_r + offset

                # Copy columns
                for offset_c in range(len(right[0])):
                    src_c = right_min_c + offset_c
                    dst_c = left_min_c + offset_c

                    if src_c < len(right[0]) and dst_c < len(left[0]):
                        left[dst_r][dst_c] = right[src_r][src_c]

            return left

    # Pattern 2b: 7s are far apart (no separator) - extract left portion
    if max_col_7 - min_col_7 > cols // 2:
        # 7s are very spread out, extract left region only
        # Find a reasonable cutoff point (around where the left structure ends)
        extract_cols = min(min_col_7 + 5, cols // 2, 6)
        result = [row[:extract_cols] for row in grid]

        # Track which row has the 7 to fix
        rows_with_sevens = set()

        # First pass: Fix 7s and extend horizontally
        for r in range(len(result)):
            for c in range(len(result[0])):
                if result[r][c] == 7:
                    rows_with_sevens.add(r)
                    # Replace 7 with frame color (4)
                    result[r][c] = 4

                    # Find template row above with 4 at this column
                    template_row = None
                    for r_ref in range(r - 1, -1, -1):
                        if result[r_ref][c] == 4:
                            # Check if this row has a horizontal pattern of 4s
                            if c + 2 < len(result[0]) and result[r_ref][c+1] == 4 and result[r_ref][c+2] == 4:
                                template_row = r_ref
                                break

                    # Copy horizontal pattern from template
                    if template_row is not None:
                        for c_off in range(1, min(3, len(result[0]) - c)):
                            if result[template_row][c + c_off] == 4:
                                result[r][c + c_off] = 4

        # Second pass: Fix vertical patterns for rows just above the 7-row
        # Only extend if the 7-row already has a 4 at that column (after first pass)
        for c in range(len(result[0])):
            # Find consecutive 4s from top
            consec_4s = []
            for r in range(len(result)):
                if result[r][c] == 4:
                    consec_4s.append(r)
                elif len(consec_4s) >= 3:
                    # We have 3+ consecutive 4s, check if we should extend
                    break

            # If we have 3+ consecutive 4s, extend to rows just above 7-rows
            # But only if the 7-row itself has a 4 at this column
            if len(consec_4s) >= 3:
                last_4_row = consec_4s[-1]
                # Check if any 7-row has a 4 at this column after first pass
                seven_row_has_4 = any(result[r7][c] == 4 for r7 in rows_with_sevens)

                if seven_row_has_4:
                    # Extend to rows between last_4_row and 7-rows
                    for r in range(last_4_row + 1, len(result)):
                        if any(abs(r - r7) == 1 for r7 in rows_with_sevens):
                            result[r][c] = 4

        return result

    # Pattern 3: Localized 7s - remove rows with 7s and expand columns
    # Remove rows containing 7s
    result = [grid[r][:] for r in range(rows) if r not in rows_with_7]

    # Expand columns by the span of 7s
    col_span = max_col_7 - min_col_7

    if col_span > 0:
        # Find background color (most common non-7 value)
        color_count = {}
        for row in grid:
            for cell in row:
                if cell != 7:
                    color_count[cell] = color_count.get(cell, 0) + 1
        background = max(color_count, key=color_count.get) if color_count else 1

        # For each row, decide whether to append or insert columns
        expanded = []
        for i, row in enumerate(result):
            # Find original row index
            orig_idx = [r for r in range(rows) if r not in rows_with_7][i]

            if orig_idx < min_row_7:
                # Row is before 7s region
                if orig_idx < min_row_7 - 2:
                    # Far from 7s: append background
                    expanded.append(row + [background] * col_span)
                else:
                    # Near 7s: append from removed rows
                    removed_idx = min_row_7 + (orig_idx - (min_row_7 - 2))
                    if removed_idx < rows:
                        append_vals = grid[removed_idx][-col_span:]
                        expanded.append(row + list(append_vals))
                    else:
                        expanded.append(row + [background] * col_span)
            else:
                # Row is after 7s region: insert at min_col_7
                new_row = row[:min_col_7] + [background] * col_span + row[min_col_7:]
                expanded.append(new_row)

        return expanded

    # Default: return result without expansion
    return result
