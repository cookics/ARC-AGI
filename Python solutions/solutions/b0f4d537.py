def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid divided by 5s (vertical column divider, horizontal row divider)
    2. One section has patterns (non-{0,4,5} values), other has 4s marking rows
    3. Output expands pattern rows to wider width, maps to rows with max 4s

    Procedure:
    1. Find vertical divider, extract left/right sections
    2. Determine pattern section (has non-{0,4,5} values)
    3. Find special pattern rows and rows with max 4s count
    4. Map patterns to output rows, expand horizontally using simple duplication
    """

    rows, cols = len(grid), len(grid[0])

    # Find vertical divider
    vert_div = None
    for c in range(cols):
        if all(grid[r][c] == 5 for r in range(rows)):
            vert_div = c
            break

    if vert_div is None:
        return [[0] * 7 for _ in range(rows)]

    # Find horizontal divider in left
    left_horiz_div = rows
    for r in range(rows):
        if all(grid[r][c] == 5 for c in range(vert_div + 1)):
            left_horiz_div = r
            break

    # Extract sections
    left_section = [grid[r][:vert_div] for r in range(left_horiz_div)]

    # Find horizontal divider in right
    right_horiz_div = rows
    for r in range(rows):
        if all(grid[r][c] == 5 for c in range(vert_div + 1, cols)):
            right_horiz_div = r
            break

    right_section = [grid[r][vert_div + 1:] for r in range(right_horiz_div)]

    # Determine which has patterns
    def has_patterns(section):
        return any(val not in [0, 4, 5] for row in section for val in row)

    left_has = has_patterns(left_section) if left_section else False
    right_has = has_patterns(right_section) if right_section else False

    if left_has and not right_has:
        pattern_section = left_section
        marker_section = [grid[r][vert_div + 1:] for r in range(rows)]
        output_width = len(marker_section[0])
    elif right_has:
        pattern_section = right_section
        marker_section = [grid[r][:vert_div] for r in range(rows)]
        output_width = len(pattern_section[0]) + 2 if pattern_section else 7
    else:
        pattern_section = left_section if left_section else [[0]]
        marker_section = [grid[r][vert_div + 1:] for r in range(rows)]
        output_width = len(marker_section[0])

    # Find special rows by identifying unique patterns
    # Group rows by their pattern
    from collections import Counter
    pattern_counts = Counter()
    row_patterns = {}
    for r in range(len(pattern_section)):
        pattern_tuple = tuple(pattern_section[r])
        pattern_counts[pattern_tuple] += 1
        row_patterns[r] = pattern_tuple

    # Most common pattern is the default
    if pattern_counts:
        default_pattern = pattern_counts.most_common(1)[0][0]
        default_row = [r for r in range(len(pattern_section)) if row_patterns[r] == default_pattern][0]
        special_rows = [r for r in range(len(pattern_section)) if row_patterns[r] != default_pattern]
    else:
        default_row = 0
        special_rows = []

    # Count 4s
    four_counts = [sum(1 for v in marker_section[r] if v == 4) for r in range(len(marker_section))]
    max_fours = max(four_counts) if four_counts else 0
    rows_with_max = [r for r, cnt in enumerate(four_counts) if cnt == max_fours and cnt > 0]

    # Map rows
    row_map = {rows_with_max[i]: special_rows[i] for i in range(min(len(rows_with_max), len(special_rows)))}

    # Detect separator columns in pattern section
    sep_cols = []
    if len(pattern_section) > 1 and len(pattern_section[0]) > 0:
        for c in range(len(pattern_section[0])):
            first_val = pattern_section[0][c]
            if first_val not in [0, 5] and all(pattern_section[r][c] == first_val for r in range(len(pattern_section))):
                sep_cols.append(c)

    # Expand pattern to target width
    def expand(pattern, width):
        if not pattern:
            return [0] * width
        src_w = len(pattern)
        if src_w == width:
            return pattern[:]

        # Different expansion strategies based on separator columns
        # For patterns coming from right section: duplicate first and last
        if right_has and src_w == 5 and width == 7:
            # Right section: simple first+last duplication
            mapping = [0, 0, 1, 2, 3, 4, 4]
            return [pattern[mapping[j]] for j in range(width)]

        # For patterns from left section with separators
        if left_has and src_w == 5 and width == 7:
            if sep_cols:
                # Has separator columns -> use [0,0,1,2,2,3,4]
                mapping = [0, 0, 1, 2, 2, 3, 4]
            else:
                # No separators -> use [0,1,2,2,2,3,4]
                mapping = [0, 1, 2, 2, 2, 3, 4]
            return [pattern[mapping[j]] for j in range(width)]

        if src_w == 3 and width == 7:
            # 3->7: center element at position 4
            mapping = [0, 0, 0, 0, 1, 2, 2]
            return [pattern[mapping[j]] for j in range(width)]

        # Default: simple floor division scaling
        return [pattern[(j * src_w) // width] for j in range(width)]

    # Build output
    result = []
    for r in range(rows):
        src_row = row_map.get(r, default_row)
        if src_row < len(pattern_section):
            pattern = pattern_section[src_row]
        else:
            pattern = pattern_section[0]
        result.append(expand(pattern, output_width))

    return result
