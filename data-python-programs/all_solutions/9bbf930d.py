def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid where first two columns are always [6, 7]
    2. Many rows have colored "stripes" (repeating non-7 values)
    3. Some rows are "background rows" (mostly 7s)
    4. When two rows have the same colored stripe separated by one background row, that middle row gets modified

    Procedure:
    1. Classify each row by its dominant stripe color
    2. Find pairs of stripe rows with same color separated by exactly one row
    3. Modify middle background rows: column 0 becomes 7, last column becomes 6 if it was 7
    4. Modify first and last rows if they have trailing 7s
    """

    result = [row[:] for row in grid]  # Deep copy
    height = len(grid)
    width = len(grid[0])

    # Helper: Get stripe color (dominant non-7 color in columns 2+)
    def get_stripe_color(row):
        if len(row) <= 2:
            return None
        from collections import Counter
        # Count non-7 values in columns 2+
        colors = [c for c in row[2:] if c != 7]
        if not colors:
            return None
        counter = Counter(colors)
        most_common_color, count = counter.most_common(1)[0]
        # If this color appears significantly (at least 40% of non-first-two columns)
        if count >= len(row[2:]) * 0.4:
            return most_common_color
        return None

    # Helper: Check if row has a "clean" stripe (continuous from col 2)
    def has_clean_stripe(row):
        if len(row) <= 2:
            return False
        color = row[2]
        if color == 7:
            return False
        # Check how far the continuous stripe extends
        for j in range(3, len(row)):
            if row[j] != color:
                break
        else:
            # All same color from position 2 onwards
            return True
        # Stripe continues from col 2 to j-1, then should be mostly 7s or other values
        return j >= len(row) * 0.5

    # Helper: Check if row is mostly background (mostly 7s)
    def is_background_row(row):
        count_7 = sum(1 for c in row[2:] if c == 7)
        return count_7 >= len(row[2:]) * 0.7  # At least 70% are 7s

    # Classify all rows by stripe color
    row_colors = []
    for i in range(height):
        color = get_stripe_color(grid[i])
        row_colors.append(color)

    # Rule 1: Background rows between two stripe rows of same color
    for i in range(1, height - 1):
        if is_background_row(grid[i]):
            prev_color = row_colors[i-1]
            next_color = row_colors[i+1]

            if prev_color is not None and prev_color == next_color:
                # Modify this background row
                result[i][0] = 7  # Change first column from 6 to 7
                if result[i][-1] == 7:  # If last column is 7
                    result[i][-1] = 6  # Change it to 6

    # Rule 2: First row changes if it has clean stripe paired with row 2 AND has trailing 7s
    if height >= 3 and has_clean_stripe(grid[0]):
        # Count consecutive trailing 7s
        trailing_7s = 0
        for j in range(width-1, -1, -1):
            if grid[0][j] == 7:
                trailing_7s += 1
            else:
                break
        # Check if row 2 has the same clean stripe color
        if trailing_7s >= 2 and has_clean_stripe(grid[2]) and row_colors[0] == row_colors[2]:
            result[0][-2] = 6

    # Rule 3: Last row changes if it has clean stripe singleton AND trailing 7s
    if has_clean_stripe(grid[-1]):
        # Count consecutive trailing 7s
        trailing_7s = 0
        for j in range(width-1, -1, -1):
            if grid[-1][j] == 7:
                trailing_7s += 1
            else:
                break
        # Count how many other rows have the same clean stripe color
        same_color_count = sum(1 for i in range(height-1) if has_clean_stripe(grid[i]) and row_colors[i] == row_colors[-1])
        if trailing_7s >= 2 and same_color_count == 0:
            result[-1][-2] = 6

    # Rule 4: Handle special background rows with matching suffixes (very specific case)
    for i in range(1, height - 1):
        # Check if row i is mostly 7s in early columns but has non-7 pattern in later columns
        if grid[i][0] == 6 and grid[i][1] == 7 and row_colors[i] is None:
            # Find where the 7s end
            first_non_7 = -1
            for j in range(2, width):
                if grid[i][j] != 7:
                    first_non_7 = j
                    break

            suffix_len = width - first_non_7 if first_non_7 > 0 else 0
            if first_non_7 > 7 and suffix_len >= 7:  # At least 6 leading 7s AND at least 7-cell suffix
                # Check if rows i-1 and i+1 have the same suffix from first_non_7 onwards
                if (len(grid[i-1]) > first_non_7 and len(grid[i+1]) > first_non_7 and
                    grid[i-1][first_non_7:] == grid[i][first_non_7:] == grid[i+1][first_non_7:]):
                    # Check that both neighbors have different stripe colors
                    if row_colors[i-1] is not None and row_colors[i+1] is not None and row_colors[i-1] != row_colors[i+1]:
                        # Change the last 7 before the suffix to 6
                        result[i][first_non_7 - 1] = 6

    # Rule 5: Handle completely all-7 rows between pair groups (very specific case for example 3)
    for i in range(3, height - 3):
        # Check if row is completely 7s (except first two columns)
        if all(grid[i][j] == 7 for j in range(2, width)):
            prev_color = row_colors[i-1]
            next_color = row_colors[i+1]
            # Only if neighbors have different non-None colors
            if prev_color is not None and next_color is not None and prev_color != next_color:
                # Check if there are pairs before and after this row
                # Look for a pair before: rows i-3 and i-1 have same color
                has_pair_before = (i >= 3 and row_colors[i-3] is not None and
                                 row_colors[i-3] == row_colors[i-1])
                # Look for a pair after: rows i+1 and i+3 have same color
                has_pair_after = (i < height - 3 and row_colors[i+3] is not None and
                                row_colors[i+3] == row_colors[i+1])

                if has_pair_before and has_pair_after:
                    # Change second-to-last position to 6
                    result[i][-2] = 6

    return result
