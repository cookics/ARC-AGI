def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid is divided into rectangular regions using 1s as dividers
    2. Each region has a dominant color (most common non-1 value)
    3. Crosses of 1s mark special regions with rare marker values
    4. Output transposes the region grid and scales it to fill the output dimensions
    5. For rectangular inputs: dimensions swap (HxW -> WxH)

    Procedure:
    1. Find all row and column boundaries where 1s cluster
    2. Extract regions between boundaries and determine dominant color for each
    3. Create a 2D grid of region colors
    4. Transpose this region grid
    5. Scale the transposed region grid to match output dimensions
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    from collections import Counter

    # Find boundaries based on 1s clustering
    row_ones = [sum(1 for c in range(cols) if grid[r][c] == 1) for r in range(rows)]
    col_ones = [sum(1 for r in range(rows) if grid[r][c] == 1) for c in range(cols)]

    # Identify rows/cols with significant 1s
    row_divs = [r for r in range(rows) if row_ones[r] >= 2]
    col_divs = [c for c in range(cols) if col_ones[c] >= 2]

    # Cluster consecutive dividers
    def cluster_dividers(divs):
        if not divs:
            return []
        clusters = []
        current = [divs[0]]
        for d in divs[1:]:
            if d - current[-1] <= 2:
                current.append(d)
            else:
                clusters.append(current)
                current = [d]
        clusters.append(current)
        return [(min(c), max(c)) for c in clusters]

    row_clusters = cluster_dividers(row_divs)
    col_clusters = cluster_dividers(col_divs)

    # Create boundaries (regions between dividers)
    row_bounds = []
    prev = 0
    for start, end in row_clusters:
        if prev < start:
            row_bounds.append(prev)
        prev = end + 1
    if prev < rows:
        row_bounds.append(prev)
    row_bounds.append(rows)

    col_bounds = []
    prev = 0
    for start, end in col_clusters:
        if prev < start:
            col_bounds.append(prev)
        prev = end + 1
    if prev < cols:
        col_bounds.append(prev)
    col_bounds.append(cols)

    # Extract region grid (dominant color in each region)
    region_grid = []
    for i in range(len(row_bounds) - 1):
        row_regions = []
        for j in range(len(col_bounds) - 1):
            r1, r2 = row_bounds[i], row_bounds[i + 1]
            c1, c2 = col_bounds[j], col_bounds[j + 1]

            # Count colors in region (excluding 1)
            counter = Counter()
            for r in range(r1, r2):
                for c in range(c1, c2):
                    if grid[r][c] != 1:
                        counter[grid[r][c]] += 1

            # Get dominant color
            if counter:
                dominant = counter.most_common(1)[0][0]
            else:
                dominant = 0
            row_regions.append(dominant)
        region_grid.append(row_regions)

    # Transpose region grid
    if region_grid and region_grid[0]:
        region_grid = list(zip(*region_grid))
    else:
        region_grid = []

    # Determine output dimensions
    out_rows = cols if rows != cols else rows
    out_cols = rows if rows != cols else cols

    # Scale region grid to output dimensions
    result = [[0] * out_cols for _ in range(out_rows)]

    if region_grid and region_grid[0]:
        num_region_rows = len(region_grid)
        num_region_cols = len(region_grid[0])

        for r in range(out_rows):
            for c in range(out_cols):
                # Map output cell to region
                region_r = r * num_region_rows // out_rows
                region_c = c * num_region_cols // out_cols
                result[r][c] = region_grid[region_r][region_c]

    return result
