def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10×10 grid with a 4×4 main block and isolated marker values
    2. Output is an 8×8 grid (with 0 borders) divided into 4 quadrants
    3. Top-left (TL) quadrant is the main 4×4 block from input
    4. Top-right (TR) quadrant is TL horizontally flipped with values remapped to TR markers
    5. Bottom-left (BL) quadrant is TL vertically flipped with values remapped to BL markers
    6. Bottom-right (BR) quadrant is BL horizontally flipped with values remapped to BR markers
    7. Markers are classified by position: TR (row<5, col>=5), BL (row>=5, col<5), BR (row>=5, col>=5)

    Procedure:
    1. Extract the 4×4 TL block from rows 1-4, cols 1-4
    2. Collect all markers and classify them as TR, BL, or BR based on position
    3. Determine TL→TR value mapping by finding row with single unique value
    4. Create TR by flipping TL horizontally and applying mapping
    5. Determine TL→BL mapping using reversed sorted mapping
    6. Create BL by flipping TL vertically and applying mapping
    7. Determine BL→BR mapping using reversed sorted mapping
    8. Create BR by flipping BL horizontally and applying mapping
    9. Assemble the output grid
    """
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]

    # Extract TL block (rows 1-4, cols 1-4)
    tl = []
    for r in range(1, 5):
        row = []
        for c in range(1, 5):
            row.append(grid[r][c])
        tl.append(row)

    # Get unique values in TL
    tl_values = set()
    for row in tl:
        for val in row:
            if val != 0:
                tl_values.add(val)

    # Collect markers
    tr_markers = set()
    bl_markers = set()
    br_markers = set()
    br_has_col5 = False  # Check if BR markers exist at col 5

    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0 and val not in tl_values:
                if r < 5 and c >= 5:
                    tr_markers.add(val)
                elif r >= 5 and c < 5:
                    bl_markers.add(val)
                elif r >= 5 and c >= 5:
                    br_markers.add(val)
                    if c == 5:
                        br_has_col5 = True

    # Determine TL→TR mapping
    # Find a row in TL with a single unique value
    tr_map = {}
    if len(tl_values) == 2 and len(tr_markers) >= 1:
        tl_vals_sorted = sorted(tl_values)
        tr_vals_sorted = sorted(tr_markers)

        # Find row with single unique value
        for row_idx, row in enumerate(tl):
            row_vals = set(v for v in row if v != 0)
            if len(row_vals) == 1:
                uniform_val = list(row_vals)[0]
                # Find marker in corresponding row
                for c in range(5, w):
                    marker_val = grid[row_idx + 1][c]
                    if marker_val != 0:
                        tr_map[uniform_val] = marker_val
                        # Map other value to other marker
                        other_tl = [v for v in tl_vals_sorted if v != uniform_val][0]
                        other_tr = [v for v in tr_vals_sorted if v != marker_val][0] if len(tr_vals_sorted) > 1 else marker_val
                        tr_map[other_tl] = other_tr
                        break
                if tr_map:
                    break

        # If no mapping found, use reversed sorted mapping
        if not tr_map and len(tr_vals_sorted) >= 2:
            tr_map = {tl_vals_sorted[0]: tr_vals_sorted[1], tl_vals_sorted[1]: tr_vals_sorted[0]}
        elif not tr_map and len(tr_vals_sorted) == 1:
            tr_map = {tl_vals_sorted[0]: tr_vals_sorted[0], tl_vals_sorted[1]: tr_vals_sorted[0]}

    # Determine TL→BL mapping (reversed sorted)
    bl_map = {}
    if len(tl_values) == 2 and len(bl_markers) >= 2:
        tl_vals_sorted = sorted(tl_values)
        bl_vals_sorted = sorted(bl_markers)
        bl_map = {tl_vals_sorted[0]: bl_vals_sorted[1], tl_vals_sorted[1]: bl_vals_sorted[0]}
    elif len(tl_values) == 2 and len(bl_markers) == 1:
        tl_vals_sorted = sorted(tl_values)
        bl_val = list(bl_markers)[0]
        bl_map = {tl_vals_sorted[0]: bl_val, tl_vals_sorted[1]: bl_val}

    # Create TR (horizontal flip of TL with mapping)
    tr = []
    for row in tl:
        new_row = []
        for c in range(3, -1, -1):  # Reverse column order
            val = row[c]
            new_row.append(tr_map.get(val, val))
        tr.append(new_row)

    # Create BL (vertical flip of TL with mapping)
    bl = []
    for r in range(3, -1, -1):  # Reverse row order
        new_row = []
        for c in range(4):
            val = tl[r][c]
            new_row.append(bl_map.get(val, val))
        bl.append(new_row)

    # Get BL values and determine BL→BR mapping
    bl_values = set()
    for row in bl:
        for val in row:
            if val != 0:
                bl_values.add(val)

    br_map = {}
    if len(bl_values) >= 2 and len(br_markers) >= 1:
        bl_vals_sorted = sorted(bl_values)
        br_vals_sorted = sorted(br_markers)

        # Check if we need to map to 0 (background) for sparse BR
        if len(br_vals_sorted) == 1 and not br_has_col5:
            # Find which BL value appears alone in a row (maps to 0)
            val_to_zero = None
            for row in bl:
                row_vals = set(v for v in row if v != 0)
                if len(row_vals) == 1:
                    val_to_zero = list(row_vals)[0]
                    break

            if val_to_zero:
                other_val = [v for v in bl_vals_sorted if v != val_to_zero][0]
                br_map = {val_to_zero: 0, other_val: br_vals_sorted[0]}
            else:
                # Default: reversed mapping with 0
                br_map = {bl_vals_sorted[0]: br_vals_sorted[0], bl_vals_sorted[1]: 0}
        elif len(br_vals_sorted) >= 2:
            br_map = {bl_vals_sorted[0]: br_vals_sorted[1], bl_vals_sorted[1]: br_vals_sorted[0]}
        else:
            br_val = br_vals_sorted[0] if br_vals_sorted else 0
            br_map = {bl_vals_sorted[0]: br_val, bl_vals_sorted[1]: br_val}

    # Create BR (horizontal flip of BL with mapping)
    br = []
    for row in bl:
        new_row = []
        for c in range(3, -1, -1):  # Reverse column order
            val = row[c]
            new_row.append(br_map.get(val, val))
        br.append(new_row)

    # Assemble output
    # Copy TL
    for r in range(4):
        for c in range(4):
            result[r + 1][c + 1] = tl[r][c]

    # Copy TR
    for r in range(4):
        for c in range(4):
            result[r + 1][c + 5] = tr[r][c]

    # Copy BL
    for r in range(4):
        for c in range(4):
            result[r + 5][c + 1] = bl[r][c]

    # Copy BR
    if br_has_col5:
        # BR uses cols 5-8
        for r in range(4):
            for c in range(4):
                result[r + 5][c + 5] = br[r][c]
    else:
        # BR uses cols 6-8 only (col 5 is separator)
        for r in range(4):
            for c in range(3):  # Only 3 columns (skip first column of BR which maps to col 5)
                result[r + 5][c + 6] = br[r][c + 1]

    return result
