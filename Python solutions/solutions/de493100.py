def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a rectangular block of 7s marking a region
    2. Output dimensions match the 7s block dimensions
    3. Data comes from two sources:
       - Rows that match the 7s block row pattern (same values outside 7s columns)
       - Transposed region (reading columns vertically) for unmatched rows

    Procedure:
    1. Find the 7s block bounds (determines output dimensions)
    2. For each row R in the 7s block:
       a. Search for a row with identical values in all non-7s columns
       b. If found, extract from that row at the 7s column positions
       c. If not found, extract from transposed region (column read)
    """

    rows, cols = len(grid), len(grid[0])

    # Find 7s bounds
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 7:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)

    height = max_row - min_row + 1
    width = max_col - min_col + 1
    result = []

    # For each row in 7s region
    for row_idx in range(min_row, max_row + 1):
        curr_row = grid[row_idx]
        best_match_row = None

        # Search for matching row outside 7s block
        for search_row in range(rows):
            # Skip if this row has 7s at the target columns
            has_7s_at_target = any(grid[search_row][c] == 7
                                   for c in range(min_col, max_col + 1))
            if has_7s_at_target:
                continue

            # Check if all non-7s, non-target columns match
            match_count = 0
            total_count = 0

            for c in range(cols):
                # Skip the 7s columns
                if min_col <= c <= max_col:
                    continue
                # Skip if either cell has a 7
                if grid[search_row][c] == 7 or curr_row[c] == 7:
                    continue

                total_count += 1
                if grid[search_row][c] == curr_row[c]:
                    match_count += 1

            # If perfect match, use this row
            if total_count > 0 and match_count == total_count:
                best_match_row = search_row
                break

        # Extract the output row
        if best_match_row is not None:
            # Found a matching row - extract from it
            row = [grid[best_match_row][min_col + j] for j in range(width)]
        else:
            # No match found - try transposed region (symmetric position)
            row = None

            # Calculate which output row this corresponds to
            output_row_idx = row_idx - min_row

            # Try reading from a column at symmetric position (right side of grid)
            # If 7s block is at cols 0-9, try reading from cols grid_width-10 to grid_width-1
            mirror_col_start = cols - max_col - 1
            mirror_col = mirror_col_start + output_row_idx

            # Try reading this column from a symmetric row range
            mirror_row_start = min_row - height

            if (0 <= mirror_col < cols and
                0 <= mirror_row_start and
                mirror_row_start + height <= rows):

                # Read column downward
                candidate = [grid[mirror_row_start + j][mirror_col] for j in range(height)]
                if all(candidate[j] != 7 for j in range(height)):
                    row = candidate

            # If that didn't work, search all horizontal segments
            if not row:
                for search_row in range(rows):
                    candidate_row = grid[search_row]

                    for start_col in range(0, cols - width + 1):
                        # Skip if overlaps with 7s columns
                        if start_col <= max_col and start_col + width > min_col:
                            continue

                        # Check if this range has no 7s
                        if any(candidate_row[start_col + j] == 7 for j in range(width)):
                            continue

                        # Found a valid candidate
                        row = [candidate_row[start_col + j] for j in range(width)]
                        break

                    if row:
                        break

            # Last resort
            if not row:
                row = [0] * width

        result.append(row)

    return result
