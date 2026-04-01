def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid divided by 5s into regions (3 rows: templates, selectors, markers)
    2. Top row contains pattern templates with different values
    3. Middle row(s) contain selector values at specific positions
    4. Bottom row contains marker colors
    5. For each value V at position (r,c) in a middle region, place pattern V at output block (r,c)
    6. Color is determined by pairing middle columns with bottom columns

    Procedure:
    1. Extract regions by splitting on separator lines of 5s
    2. Extract patterns from top row regions
    3. For each middle region, find positions of each value
    4. Determine pairing between middle and bottom columns
    5. Place patterns at output positions based on middle region positions
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Remove trailing all-zero rows
    while rows > 0 and all(grid[rows-1][c] == 0 for c in range(cols)):
        rows -= 1

    if rows == 0:
        return [[0]]

    # Find separator rows and columns
    sep_rows = []
    for r in range(rows):
        count_5 = sum(1 for c in range(cols) if grid[r][c] == 5)
        if count_5 > cols // 2:
            sep_rows.append(r)

    sep_cols = []
    for c in range(cols):
        count_5 = sum(1 for r in range(rows) if grid[r][c] == 5)
        if count_5 > rows // 2:
            sep_cols.append(c)

    # Extract non-separator regions
    row_ranges = []
    start = 0
    for sep in sep_rows:
        if start < sep:
            row_ranges.append((start, sep))
        start = sep + 1
    if start < rows:
        row_ranges.append((start, rows))

    col_ranges = []
    start = 0
    for sep in sep_cols:
        if start < sep:
            col_ranges.append((start, sep))
        start = sep + 1
    if start < cols:
        col_ranges.append((start, cols))

    # Extract all regions
    regions = []
    for r_start, r_end in row_ranges:
        row_regions = []
        for c_start, c_end in col_ranges:
            region = [[grid[r][c] for c in range(c_start, c_end)] for r in range(r_start, r_end)]
            row_regions.append(region)
        regions.append(row_regions)

    if not regions or not regions[0] or len(regions) < 2:
        return [[]]

    def dominant_val(region):
        """Get most common non-zero, non-5 value"""
        vals = [v for row in region for v in row if v not in (0, 5)]
        if not vals:
            return 0
        return Counter(vals).most_common(1)[0][0]

    # Extract patterns from top row (store value -> pattern mapping)
    patterns_by_value = {}
    for col_idx, region in enumerate(regions[0]):
        val = dominant_val(region)
        if val > 0:
            patterns_by_value[val] = [[v for v in row] for row in region]

    # Extract markers from bottom row (col_idx -> marker)
    markers_by_col = {}
    for col_idx, region in enumerate(regions[-1]):
        marker = dominant_val(region)
        if marker > 0:
            markers_by_col[col_idx] = marker

    # Collect values from middle regions (all regions between top and bottom) with their positions
    middle_values = {}  # {value: [(region_col, pos_r, pos_c), ...]}
    # Process all middle rows (between first and last)
    for region_row_idx in range(1, len(regions) - 1):
        for region_col, region in enumerate(regions[region_row_idx]):
            for r in range(len(region)):
                for c in range(len(region[0])):
                    val = region[r][c]
                    if val not in (0, 5):
                        if val not in middle_values:
                            middle_values[val] = []
                        middle_values[val].append((region_col, r, c))

    # Determine value-to-marker pairing
    # Rule: For middle col j with value V, find top col k with pattern V, use marker from bottom col k
    # First, create value-to-top-col mapping
    value_to_top_col = {}
    for col_idx, region in enumerate(regions[0]):
        val = dominant_val(region)
        if val > 0:
            value_to_top_col[val] = col_idx

    # Now create value-to-marker mapping via top column
    value_to_marker = {}
    for val in middle_values.keys():
        if val in value_to_top_col:
            top_col = value_to_top_col[val]
            if top_col in markers_by_col:
                value_to_marker[val] = markers_by_col[top_col]
            else:
                # Fallback: use any available marker
                for marker in markers_by_col.values():
                    if marker not in value_to_marker.values():
                        value_to_marker[val] = marker
                        break

    # Determine output dimensions based on maximum positions in middle values
    block_h = len(regions[0][0]) if regions[0] and regions[0][0] else 3
    block_w = len(regions[0][0][0]) if regions[0] and regions[0][0] and regions[0][0][0] else 3

    # Find max row and col positions from middle values
    max_pos_r = 0
    max_pos_c = 0
    for positions in middle_values.values():
        for _, pos_r, pos_c in positions:
            max_pos_r = max(max_pos_r, pos_r)
            max_pos_c = max(max_pos_c, pos_c)

    num_block_rows = max_pos_r + 1 if middle_values else len(regions[0])
    num_block_cols = max_pos_c + 1 if middle_values else len(regions[0])

    # Ensure valid dimensions
    if num_block_rows == 0 or num_block_cols == 0 or block_h == 0 or block_w == 0:
        return [[0]]

    out_h = num_block_rows * block_h
    out_w = num_block_cols * block_w
    output = [[0] * out_w for _ in range(out_h)]

    # Place patterns in output based on middle region positions
    for val, positions in middle_values.items():
        if val not in patterns_by_value:
            continue
        pattern = patterns_by_value[val]
        marker = value_to_marker.get(val, val)

        for region_col, pos_r, pos_c in positions:
            # Place pattern at output block (pos_r, pos_c)
            for dr in range(min(block_h, len(pattern))):
                for dc in range(min(block_w, len(pattern[0]))):
                    out_r = pos_r * block_h + dr
                    out_c = pos_c * block_w + dc
                    if 0 <= out_r < out_h and 0 <= out_c < out_w:
                        if pattern[dr][dc] not in (0, 5):
                            output[out_r][out_c] = marker

    return output
