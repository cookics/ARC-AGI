def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a colored block (9s and 1s) at rows 0-2
    2. This block is removed and each column is extracted as a vertical strip
    3. Each strip is placed at a different row position (same column, different rows)
    4. Placement is determined by the positions of 5s patterns

    Procedure:
    1. Extract colored block columns from rows 0-2
    2. For each column, find where to place it based on 5s in that column
    3. Place the vertical strip at the determined position
    4. Clear the original colored block area
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find colored block boundaries (rows 0-2)
    colored_cols = []
    for j in range(cols):
        if grid[0][j] not in [7, 5] or grid[1][j] not in [7, 5] or grid[2][j] not in [7, 5]:
            colored_cols.append(j)

    if not colored_cols:
        return result

    # Extract each column's vertical strip
    column_strips = {}
    for col in colored_cols:
        strip = [grid[0][col], grid[1][col], grid[2][col]]
        column_strips[col] = strip

    # Clear original colored block area
    for i in range(3):
        for j in colored_cols:
            result[i][j] = 7

    # For each column, find placement
    placements = {}

    for col in colored_cols:
        # Find 5s in this column (after row 2)
        fives_in_col = []
        for i in range(3, rows):
            if grid[i][col] == 5:
                fives_in_col.append(i)

        # Determine placement row
        if fives_in_col:
            # Check if there are multiple separate clusters of 5s
            # If last 5 is far from first 5, use last 5 instead
            first_five = fives_in_col[0]
            last_five = fives_in_col[-1]

            # If last 5 is much later (4+ rows away), use it instead
            if last_five >= 12 and last_five - first_five >= 4:
                target_five = last_five
            else:
                target_five = first_five

            # Ignore very early 5s (rows 3-5) - treat as if no 5s
            if target_five <= 5:
                placements[col] = None
            elif target_five >= 14:
                # For very late 5s, place right before them
                place_row = target_five - 1
                placements[col] = place_row
            elif target_five >= 12:
                # For late 5s, place closer to them
                place_row = target_five - 2
                placements[col] = place_row
            elif target_five >= 8:
                place_row = target_five - 3
                placements[col] = place_row
            else:
                place_row = max(3, target_five - 3)
                placements[col] = place_row
        else:
            # No 5s in this column - will place based on global pattern
            placements[col] = None

    # For columns without placement, find row with most 5s
    if None in placements.values():
        row_five_counts = {}
        for i in range(3, rows):
            count = sum(1 for j in range(cols) if grid[i][j] == 5)
            row_five_counts[i] = count

        # Find all rows with maximum 5s
        max_count = max(row_five_counts.values()) if row_five_counts else 0
        candidate_rows = [r for r, c in row_five_counts.items() if c == max_count]

        # Use different candidate rows for different columns
        unplaced_cols = [c for c in colored_cols if placements[c] is None]
        for idx, col in enumerate(unplaced_cols):
            # Use last candidate row for placement (highest row number)
            if candidate_rows:
                best_row = candidate_rows[-1] if idx == 0 else candidate_rows[-1]
                placements[col] = min(rows - 3, best_row)
            else:
                placements[col] = 3

    # Group consecutive columns with same placement
    col_groups = []
    i = 0
    while i < len(colored_cols):
        group = [colored_cols[i]]
        placement = placements[colored_cols[i]]
        j = i + 1
        while j < len(colored_cols) and placements[colored_cols[j]] == placement:
            group.append(colored_cols[j])
            j += 1
        col_groups.append((group, placement))
        i = j

    # Check which columns have unique patterns
    pattern_counts = {}
    for col, strip in column_strips.items():
        pattern = tuple(strip)
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Place each group with column shift if needed
    for group, place_row in col_groups:
        for col in group:
            strip = column_strips[col]
            pattern = tuple(strip)
            is_unique_pattern = (pattern_counts[pattern] == 1)

            # Check if placing at this column would overwrite important 5s
            target_col = col
            has_conflict = False
            for offset in range(len(strip)):
                target_row = place_row + offset
                if 0 <= target_row < rows and grid[target_row][col] == 5:
                    has_conflict = True
                    break

            # Only shift if column has unique pattern and there's a conflict
            if has_conflict and is_unique_pattern:
                # Try col + 1 first
                if col + 1 < cols:
                    target_col = col + 1

            # Place the strip
            for offset, val in enumerate(strip):
                target_row = place_row + offset
                if 0 <= target_row < rows and 0 <= target_col < cols:
                    result[target_row][target_col] = val

                    # Clear adjacent 5s to the right (up to 2 cells)
                    for clear_offset in range(1, 3):
                        clear_col = target_col + clear_offset
                        if (0 <= clear_col < cols and
                            grid[target_row][clear_col] == 5 and
                            result[target_row][clear_col] == 5):
                            result[target_row][clear_col] = 7
                        else:
                            break  # Stop if not a 5

    return result
