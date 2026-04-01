def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large sparse grid with two distinct structures: a rectangular frame and a pattern of 8s
    2. The frame consists of four borders (top, bottom, left, right) each with uniform color
    3. The 8s pattern is located elsewhere in the grid, forming an irregular shape
    4. Output is a compact rectangle where frame borders form the outer edges
    5. Interior cells are filled based on the 8s pattern: where 8s touch exactly one frame edge, use that edge's color; interior 8s remain 8; empty positions remain 0

    Procedure:
    1. Find the four frame borders by scanning for horizontal/vertical lines of uniform non-zero, non-8 values
    2. Extract the bounding box of all 8s in the grid to get the 8s pattern
    3. Create output grid with dimensions based on frame size plus corner padding
    4. Fill output borders: top and bottom rows with frame colors flanked by 0s
    5. Fill output sides: left and right columns with frame colors
    6. Fill interior by overlaying the 8s pattern and applying transformation rules based on edge proximity
    """

    rows, cols = len(grid), len(grid[0])

    # Find frame borders
    top_border = None
    bottom_border = None
    left_border = None
    right_border = None

    # Find top border - horizontal line of same non-zero, non-8 values
    for r in range(rows):
        line_values = []
        start_col = None
        end_col = None
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 8:
                if start_col is None:
                    start_col = c
                line_values.append(grid[r][c])
                end_col = c
            elif len(line_values) > 0:
                break

        if len(line_values) >= 3 and len(set(line_values)) == 1:
            top_border = {
                "row": r,
                "start_col": start_col,
                "end_col": end_col,
                "values": line_values,
                "color": line_values[0],
            }
            break

    # Find bottom border
    for r in range(rows - 1, -1, -1):
        line_values = []
        start_col = None
        end_col = None
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 8:
                if start_col is None:
                    start_col = c
                line_values.append(grid[r][c])
                end_col = c
            elif len(line_values) > 0:
                break

        if len(line_values) >= 3 and len(set(line_values)) == 1:
            bottom_border = {
                "row": r,
                "start_col": start_col,
                "end_col": end_col,
                "values": line_values,
                "color": line_values[0],
            }
            break

    # Find left border - vertical line
    for c in range(cols):
        line_values = []
        start_row = None
        end_row = None
        for r in range(rows):
            if grid[r][c] != 0 and grid[r][c] != 8:
                if start_row is None:
                    start_row = r
                line_values.append(grid[r][c])
                end_row = r

        # Only consider if we have enough values and they're all the same
        if len(line_values) >= 3 and len(set(line_values)) == 1:
            left_border = {
                "col": c,
                "start_row": start_row,
                "end_row": end_row,
                "values": line_values,
                "color": line_values[0],
            }
            break

    # Find right border
    for c in range(cols - 1, -1, -1):
        line_values = []
        start_row = None
        end_row = None
        for r in range(rows):
            if grid[r][c] != 0 and grid[r][c] != 8:
                if start_row is None:
                    start_row = r
                line_values.append(grid[r][c])
                end_row = r

        if len(line_values) >= 3 and len(set(line_values)) == 1:
            right_border = {
                "col": c,
                "start_row": start_row,
                "end_row": end_row,
                "values": line_values,
                "color": line_values[0],
            }
            break

    assert top_border and bottom_border and left_border and right_border

    # Extract 8s pattern
    min_r, max_r = rows, -1
    min_c, max_c = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Extract 8s pattern as sub-grid
    eights_grid = []
    if min_r <= max_r and min_c <= max_c:
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(grid[r][c] if grid[r][c] == 8 else 0)
            eights_grid.append(row)

    # Create output grid
    frame_width = len(top_border["values"])
    interior_height = len(left_border["values"])

    output_width = frame_width + 2  # +2 for 0-borders on left/right
    output_height = interior_height + 2  # +2 for 0-borders on top/bottom

    result = [[0 for _ in range(output_width)] for _ in range(output_height)]

    # Fill borders
    # Top row: [0, top_values..., 0]
    for i, val in enumerate(top_border["values"]):
        result[0][i + 1] = val

    # Bottom row: [0, bottom_values..., 0]
    for i, val in enumerate(bottom_border["values"]):
        result[output_height - 1][i + 1] = val

    # Interior rows: [left_color, interior..., right_color]
    for r in range(1, output_height - 1):
        result[r][0] = left_border["color"]
        result[r][output_width - 1] = right_border["color"]

    # Fill interior with combination of frame colors and 8s pattern
    interior_width = frame_width

    for r in range(interior_height):
        for c in range(interior_width):
            interior_row = r + 1
            interior_col = c + 1

            # Get value from 8s pattern
            if eights_grid and r < len(eights_grid) and c < len(eights_grid[0]):
                eights_value = eights_grid[r][c]
            else:
                eights_value = 0

            # Apply the transformation rule:
            # - If 8s_pattern = 0, output = 0
            # - If 8s_pattern = 8:
            #   - If position is interior (not on edge), output = 8
            #   - If position is on exactly one edge, output = frame color
            #   - If position is on corner (multiple edges), output = 8

            if eights_value == 0:
                value = 0
            else:  # eights_value == 8
                # Check which edges this position touches
                on_top = r == 0
                on_bottom = r == interior_height - 1
                on_left = c == 0
                on_right = c == interior_width - 1

                edge_count = sum([on_top, on_bottom, on_left, on_right])

                if edge_count == 0:
                    # Interior position
                    value = 8
                elif edge_count == 1:
                    # On exactly one edge - use corresponding frame color
                    if on_top:
                        value = top_border["color"]
                    elif on_bottom:
                        value = bottom_border["color"]
                    elif on_left:
                        value = left_border["color"]
                    elif on_right:
                        value = right_border["color"]
                else:
                    # Corner (multiple edges) - use 8
                    value = 8

            result[interior_row][interior_col] = value

    return result
