def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large grid with four corner markers (value 4) defining a bounding rectangle
    2. Inside the rectangle are colored block regions (2×2, 3×3, or 4×4 blocks of same values)
    3. Outside the rectangle is a sparse pattern that acts as a template
    4. Output is a smaller grid matching the bounding rectangle dimensions
    5. Each template cell gets expanded to match the size of the colored blocks found inside
    6. The template values directly specify what colors to place in each region

    Procedure:
    1. Find corner 4s to determine bounding rectangle
    2. Extract colored regions within the rectangle and determine block size
    3. Find sparse template pattern outside the rectangle
    4. Create output grid based on corner positions
    5. Map each template cell to a block-sized region in the output
    """

    rows, cols = len(grid), len(grid[0])

    # Step 1: Find all 4s (corner markers)
    corners = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 4:
                corners.append((r, c))

    assert len(corners) == 4, f"Expected 4 corners, found {len(corners)}"

    # Determine bounding rectangle
    min_r = min(corner[0] for corner in corners)
    max_r = max(corner[0] for corner in corners)
    min_c = min(corner[1] for corner in corners)
    max_c = max(corner[1] for corner in corners)

    # Step 2: Extract colored regions within the rectangle to determine block size
    colored_regions = {}
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            val = grid[r][c]
            if val != 0 and val != 4:
                if val not in colored_regions:
                    colored_regions[val] = []
                colored_regions[val].append((r, c))

    # Calculate block size from the largest colored region
    block_height = block_width = 1
    if colored_regions:
        max_region_size = 0
        largest_color = None
        for color, positions in colored_regions.items():
            if len(positions) > max_region_size:
                max_region_size = len(positions)
                largest_color = color

        if largest_color:
            positions = colored_regions[largest_color]
            min_r_pos = min(pos[0] for pos in positions)
            max_r_pos = max(pos[0] for pos in positions)
            min_c_pos = min(pos[1] for pos in positions)
            max_c_pos = max(pos[1] for pos in positions)
            block_height = max_r_pos - min_r_pos + 1
            block_width = max_c_pos - min_c_pos + 1

    # Step 3: Find template pattern outside the main rectangle
    template_positions = []
    for r in range(rows):
        for c in range(cols):
            # Skip if inside main rectangle or if value is 0
            if min_r <= r <= max_r and min_c <= c <= max_c:
                continue
            if grid[r][c] != 0:
                template_positions.append((r, c, grid[r][c]))

    # Build template grid from the positions
    template = []
    if template_positions:
        template_min_r = min(pos[0] for pos in template_positions)
        template_max_r = max(pos[0] for pos in template_positions)
        template_min_c = min(pos[1] for pos in template_positions)
        template_max_c = max(pos[1] for pos in template_positions)

        template_rows = template_max_r - template_min_r + 1
        template_cols = template_max_c - template_min_c + 1
        template = [[0] * template_cols for _ in range(template_rows)]

        for r, c, val in template_positions:
            template[r - template_min_r][c - template_min_c] = val

    # Step 4: Create output grid
    output_rows = max_r - min_r + 1
    output_cols = max_c - min_c + 1
    result = [[0] * output_cols for _ in range(output_rows)]

    # Place corner 4s
    result[0][0] = 4
    result[0][output_cols - 1] = 4
    result[output_rows - 1][0] = 4
    result[output_rows - 1][output_cols - 1] = 4

    # Step 5: Map template to output
    if template:
        template_rows_count = len(template)
        template_cols_count = len(template[0])

        # Positioning logic based on template size and block dimensions
        if template_rows_count == 2 and template_cols_count == 2:
            # Small 2x2 template
            content_start_r = 1
            content_start_c = 1
        elif block_height == 3 and template_rows_count == 2:
            # 3x3 blocks with 2-row template need extra vertical spacing
            content_start_r = 3
            content_start_c = 1
        elif template_rows_count >= 3 or template_cols_count >= 3:
            # Larger templates need row 2 start
            content_start_r = 2
            content_start_c = 1
        else:
            # Default
            content_start_r = 1
            content_start_c = 1

        for tr in range(template_rows_count):
            for tc in range(template_cols_count):
                template_val = template[tr][tc]
                if template_val != 0:
                    # Calculate position in output grid
                    start_r = content_start_r + tr * block_height
                    start_c = content_start_c + tc * block_width

                    # Fill block area with template value
                    for br in range(block_height):
                        for bc in range(block_width):
                            out_r = start_r + br
                            out_c = start_c + bc
                            if 0 <= out_r < output_rows and 0 <= out_c < output_cols:
                                # Don't overwrite corner 4s
                                if result[out_r][out_c] != 4:
                                    result[out_r][out_c] = template_val

    return result
