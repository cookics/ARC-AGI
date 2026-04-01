def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Horizontal line completion: partial horizontal lines of 8s get completed with 1s
    2. Checkerboard pattern completion: partial checkerboard patterns get completed with 1s
    3. Group completion: groups of scattered 8s get completion lines added in nearby empty rows

    Procedure:
    1. Find reference patterns (complete horizontal lines, checkerboard patterns)
    2. Complete partial horizontal lines with 1s
    3. Complete partial checkerboard patterns with 1s
    4. For scattered 8s groups, add completion lines in nearby empty rows
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Strategy 1: Complete horizontal lines
    complete_horizontal_line = None
    for r in range(rows):
        if all(x == 8 for x in grid[r]):
            complete_horizontal_line = grid[r]
            break

    if complete_horizontal_line:
        # Complete partial horizontal lines
        for r in range(rows):
            row = grid[r]
            if any(x == 8 for x in row) and not all(x == 8 for x in row):
                # Check if it's a partial horizontal line starting from beginning
                consecutive_8s = 0
                for c in range(cols):
                    if row[c] == 8:
                        consecutive_8s += 1
                    else:
                        break

                if consecutive_8s > 0:
                    # Complete the line with 1s
                    for c in range(consecutive_8s, cols):
                        result[r][c] = 1

    # Strategy 2: Complete checkerboard patterns
    elif has_checkerboard_pattern(grid):
        # Find the reference checkerboard pattern
        checkerboard_start = find_checkerboard_reference(grid)

        if checkerboard_start is not None:
            # Complete partial checkerboard patterns
            for r in range(rows - 1):
                if is_partial_checkerboard(grid[r], grid[r + 1]):
                    # Find where the pattern stops
                    stop_col = find_checkerboard_stop(grid[r], grid[r + 1])
                    if stop_col < cols:
                        # Complete the pattern with alternating 1s and 0s (using 1s where pattern expects non-zero)
                        for c in range(stop_col, cols):
                            expected_val1 = 8 if c % 2 == 0 else 0
                            expected_val2 = 0 if c % 2 == 0 else 8

                            if result[r][c] == 0 and expected_val1 == 8:
                                result[r][c] = 1
                            if result[r + 1][c] == 0 and expected_val2 == 8:
                                result[r + 1][c] = 1

    # Strategy 3: Handle scattered groups (Example 2)
    else:
        # Find reference horizontal line (longest line of 8s)
        reference_line = None
        max_consecutive = 0

        for r in range(rows):
            consecutive = find_longest_consecutive_8s(grid[r])
            if consecutive[1] > max_consecutive:
                max_consecutive = consecutive[1]
                reference_line = (
                    r,
                    consecutive[0],
                    consecutive[0] + consecutive[1] - 1,
                )

        if reference_line:
            ref_row, ref_start, ref_end = reference_line

            # Find groups of scattered 8s and add completion lines
            groups = find_scattered_groups(grid)

            for group_start, group_end in groups:
                # Find next empty row after this group
                empty_row = None
                for r in range(group_end + 1, rows):
                    if all(x == 0 for x in grid[r]):
                        empty_row = r
                        break

                if empty_row is not None:
                    # Determine completion line position based on group pattern
                    completion_start = determine_completion_start(
                        grid, group_start, group_end, ref_start
                    )

                    # Add completion line
                    for c in range(completion_start, cols):
                        result[empty_row][c] = 1

    return result


def has_checkerboard_pattern(grid):
    """Check if the grid contains a checkerboard pattern"""
    for r in range(len(grid) - 1):
        if is_checkerboard_pattern(grid[r], grid[r + 1]):
            return True
    return False


def find_checkerboard_reference(grid):
    """Find the starting row of a complete checkerboard pattern"""
    for r in range(len(grid) - 1):
        if is_checkerboard_pattern(grid[r], grid[r + 1]):
            return r
    return None


def is_checkerboard_pattern(row1, row2):
    """Check if two rows form a checkerboard pattern"""
    for c in range(len(row1)):
        expected_val1 = 8 if c % 2 == 0 else 0
        expected_val2 = 0 if c % 2 == 0 else 8
        if row1[c] != expected_val1 or row2[c] != expected_val2:
            return False
    return True


def is_partial_checkerboard(row1, row2):
    """Check if two rows form a partial checkerboard pattern"""
    has_pattern = False
    has_incomplete = False

    for c in range(len(row1)):
        expected_val1 = 8 if c % 2 == 0 else 0
        expected_val2 = 0 if c % 2 == 0 else 8

        if row1[c] == expected_val1 and row2[c] == expected_val2:
            has_pattern = True
        elif row1[c] == 0 and row2[c] == 0:
            if has_pattern:  # Only incomplete if we've seen some pattern
                has_incomplete = True
            continue
        else:
            return False  # Invalid pattern

    return has_pattern and has_incomplete


def find_checkerboard_stop(row1, row2):
    """Find where the checkerboard pattern stops"""
    for c in range(len(row1)):
        expected_val1 = 8 if c % 2 == 0 else 0
        expected_val2 = 0 if c % 2 == 0 else 8

        if row1[c] != expected_val1 or row2[c] != expected_val2:
            return c
    return len(row1)


def find_longest_consecutive_8s(row):
    """Find the longest consecutive sequence of 8s in a row"""
    max_length = 0
    max_start = 0
    current_length = 0
    current_start = 0

    for i, val in enumerate(row):
        if val == 8:
            if current_length == 0:
                current_start = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start = current_start
            current_length = 0

    # Check final sequence
    if current_length > max_length:
        max_length = current_length
        max_start = current_start

    return (max_start, max_length)


def find_scattered_groups(grid):
    """Find groups of rows that contain scattered 8s"""
    groups = []
    current_group_start = None

    for r in range(len(grid)):
        has_8s = any(x == 8 for x in grid[r])

        if has_8s:
            if current_group_start is None:
                current_group_start = r
        else:
            if current_group_start is not None:
                groups.append((current_group_start, r - 1))
                current_group_start = None

    # Handle final group
    if current_group_start is not None:
        groups.append((current_group_start, len(grid) - 1))

    return groups


def determine_completion_start(grid, group_start, group_end, ref_start):
    """Determine where the completion line should start based on the group pattern"""
    # Find the rightmost 8 in the group
    rightmost_8 = 0
    for r in range(group_start, group_end + 1):
        for c in range(len(grid[r]) - 1, -1, -1):
            if grid[r][c] == 8:
                rightmost_8 = max(rightmost_8, c)
                break

    # Start completion line right after the rightmost 8
    return rightmost_8 + 1
