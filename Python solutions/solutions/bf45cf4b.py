def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background color and two distinct non-background rectangular regions
    2. Output is a tiled grid where one region acts as a mask and the other as a template
    3. The region appearing first (lower row position) serves as the mask/structure
    4. The region appearing later serves as the template to be tiled
    5. Output dimensions = (mask height × template height) by (mask width × template width)
    6. Each tile in output corresponds to a cell in the mask region

    Procedure:
    1. Find background color (most frequent value in grid)
    2. Identify all non-background cells and their positions
    3. Split non-background cells into two distinct rectangular regions using spatial clustering
    4. Extract rectangular patterns from both regions
    5. Determine which region is mask (earlier/upper) vs template (later/lower) based on row position
    6. Create output grid by tiling: for each mask cell, if non-background place template tile, else place background
    """
    from collections import Counter

    # Find background color (most frequent)
    all_values = [val for row in grid for val in row]
    background = Counter(all_values).most_common(1)[0][0]

    # Find all non-background cells
    non_bg_cells = []
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                non_bg_cells.append((r, c))

    assert len(non_bg_cells) > 0, "Must have non-background cells"

    # Find distinct rectangular regions using spatial clustering
    # Use simple k-means-like clustering to separate into 2 regions

    if len(non_bg_cells) < 2:
        return grid

    # Find the bounding box of all non-background cells
    min_r = min(r for r, c in non_bg_cells)
    max_r = max(r for r, c in non_bg_cells)
    min_c = min(c for r, c in non_bg_cells)
    max_c = max(c for r, c in non_bg_cells)

    # Simple heuristic: split by the dimension with larger spread
    if (max_r - min_r) >= (max_c - min_c):
        # Split by row
        mid_r = (min_r + max_r) // 2
        region1_cells = [(r, c) for r, c in non_bg_cells if r <= mid_r]
        region2_cells = [(r, c) for r, c in non_bg_cells if r > mid_r]
    else:
        # Split by column
        mid_c = (min_c + max_c) // 2
        region1_cells = [(r, c) for r, c in non_bg_cells if c <= mid_c]
        region2_cells = [(r, c) for r, c in non_bg_cells if c > mid_c]

    # Ensure both regions have cells
    if len(region1_cells) == 0 or len(region2_cells) == 0:
        # Fallback: just split roughly in half
        mid_idx = len(non_bg_cells) // 2
        region1_cells = non_bg_cells[:mid_idx]
        region2_cells = non_bg_cells[mid_idx:]

    # Extract rectangular patterns from regions
    def extract_pattern(region_cells):
        min_r = min(r for r, c in region_cells)
        max_r = max(r for r, c in region_cells)
        min_c = min(c for r, c in region_cells)
        max_c = max(c for r, c in region_cells)

        pattern = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(grid[r][c])
            pattern.append(row)
        return pattern

    region1_pattern = extract_pattern(region1_cells)
    region2_pattern = extract_pattern(region2_cells)

    # Determine which region is A and which is the template
    # Use the region that appears later (higher minimum row) as the template
    region1_min_r = min(r for r, c in region1_cells)
    region2_min_r = min(r for r, c in region2_cells)

    if region1_min_r < region2_min_r:
        region_a_pattern = region1_pattern
        template = region2_pattern
    else:
        region_a_pattern = region2_pattern
        template = region1_pattern

    # Determine output dimensions
    region_a_h = len(region_a_pattern)
    region_a_w = len(region_a_pattern[0])
    template_h = len(template)
    template_w = len(template[0])

    output_h = region_a_h * template_h
    output_w = region_a_w * template_w

    # Create output
    result = []
    for r in range(output_h):
        row = []
        for c in range(output_w):
            # Determine which tile this cell belongs to
            tile_r = r // template_h
            tile_c = c // template_w

            # Check the corresponding cell in region A
            region_a_cell = region_a_pattern[tile_r][tile_c]

            if region_a_cell == background:
                # Use background
                row.append(background)
            else:
                # Use template
                template_r = r % template_h
                template_c = c % template_w
                row.append(template[template_r][template_c])
        result.append(row)

    return result
