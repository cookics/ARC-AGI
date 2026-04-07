def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 13×13 grid with background value 2
    2. Contains multiple rectangular regions marked by non-2 values
    3. Regions are extracted based on their bounding boxes
    4. Regions are stacked vertically, with padding on left or right based on position
    5. Each region's center position determines padding side

    Procedure:
    1. Find all connected components of non-2 values using 4-connectivity
    2. Extract bounding box for each component
    3. Sort regions by vertical position (row)
    4. Find maximum width across all regions
    5. Stack regions vertically, padding to maximum width based on position
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])

    # Find connected components using 4-connectivity
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] != 2:
                    visited[nr][nc] = True
                    cells.append((nr, nc))
                    queue.append((nr, nc))

        return cells

    # Find all components
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 2:
                comp = bfs(r, c)
                components.append(comp)

    if not components:
        return [[2]]

    # Extract bounding boxes and regions
    regions = []
    for comp in components:
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)

        # Extract region
        region_data = []
        for r in range(min_r, max_r + 1):
            region_data.append([grid[r][c] for c in range(min_c, max_c + 1)])

        regions.append({
            'data': region_data,
            'row': min_r,
            'col': min_c,
            'height': max_r - min_r + 1,
            'width': max_c - min_c + 1
        })

    # Sort by vertical position (top to bottom), then horizontal
    regions.sort(key=lambda r: (r['row'], r['col']))

    # Decide on arrangement strategy
    if len(regions) == 1:
        # Single region: return as-is
        return regions[0]['data']
    elif len(regions) == 2:
        # Two regions: check if they overlap vertically
        r1_bottom = regions[0]['row'] + regions[0]['height'] - 1
        r2_top = regions[1]['row']

        if r2_top > r1_bottom:
            # No overlap: stack vertically with padding
            max_width = max(r['width'] for r in regions)
            result = []
            mid_col = cols / 2

            for region in regions:
                region_center_col = region['col'] + region['width'] / 2
                pad_on_left = region_center_col > mid_col

                for row in region['data']:
                    padding_needed = max_width - len(row)
                    if pad_on_left:
                        padded_row = [2] * padding_needed + row
                    else:
                        padded_row = row + [2] * padding_needed
                    result.append(padded_row)
            return result

    # For 3+ regions or overlapping regions: use grid packing
    # Pack regions into output rows using greedy algorithm
    def regions_overlap(r1, r2):
        r1_bottom = r1['row'] + r1['height'] - 1
        r2_bottom = r2['row'] + r2['height'] - 1
        return not (r1_bottom < r2['row'] or r2_bottom < r1['row'])

    # Sort regions by row start (descending) for greedy packing
    sorted_regions = sorted(regions, key=lambda r: r['row'], reverse=True)

    # Pack into output rows
    output_rows = []
    for region in sorted_regions:
        # Try to add to an existing row
        placed = False
        for row in output_rows:
            # Check if this region overlaps with any region in this row
            if not any(regions_overlap(region, r) for r in row):
                row.append(region)
                placed = True
                break
        # If not placed, create a new row
        if not placed:
            output_rows.append([region])

    # Sort regions within each row by column position
    for row in output_rows:
        row.sort(key=lambda r: r['col'])

    # Assign rows (first packed row becomes top row)
    top_row_regions = output_rows[0] if len(output_rows) > 0 else []
    bottom_row_regions = output_rows[1] if len(output_rows) > 1 else []

    # Build the grid
    max_top_height = max((r['height'] for r in top_row_regions), default=0)
    max_bottom_height = max((r['height'] for r in bottom_row_regions), default=0)

    # Calculate total width needed for each row
    top_total_width = sum(r['width'] for r in top_row_regions)
    bottom_total_width = sum(r['width'] for r in bottom_row_regions)
    max_total_width = max(top_total_width, bottom_total_width)

    # Determine number of columns: use max of top and bottom row counts
    num_cols = max(len(top_row_regions), len(bottom_row_regions))
    col_widths = [0] * num_cols

    # Calculate column widths for top row
    for i, region in enumerate(top_row_regions):
        col_widths[i] = region['width']

    # For each bottom region, find which column it should go in
    # based on horizontal position in the input
    bottom_region_cols = []
    for bottom_region in bottom_row_regions:
        # Find the column index where this region should go
        # Place it in the column of the top region it's closest to
        best_col = 0
        if len(top_row_regions) > 0:
            # Find closest top region by column position
            min_dist = float('inf')
            for col_idx, top_region in enumerate(top_row_regions):
                dist = abs(bottom_region['col'] - top_region['col'])
                if dist < min_dist:
                    min_dist = dist
                    best_col = col_idx
        bottom_region_cols.append(best_col)
        # Update column width to accommodate this region
        col_widths[best_col] = max(col_widths[best_col], bottom_region['width'])

    # Add padding column if bottom row is wider than top row
    if bottom_total_width > top_total_width:
        # Add extra column(s) for padding
        total_assigned = sum(col_widths)
        if total_assigned < max_total_width:
            col_widths.append(max_total_width - total_assigned)
            num_cols += 1

    result = []

    # Build top rows
    for row_idx in range(max_top_height):
        result_row = []
        for col_idx in range(num_cols):
            if col_idx < len(top_row_regions):
                region = top_row_regions[col_idx]
                if row_idx < region['height']:
                    # Add this row from the region
                    result_row.extend(region['data'][row_idx])
                    # Pad to column width
                    padding = col_widths[col_idx] - region['width']
                    result_row.extend([2] * padding)
                else:
                    # Pad entire column
                    result_row.extend([2] * col_widths[col_idx])
            else:
                # Pad entire column
                result_row.extend([2] * col_widths[col_idx])
        result.append(result_row)

    # Build bottom rows
    for row_idx in range(max_bottom_height):
        result_row = []
        for col_idx in range(num_cols):
            # Find if any bottom region belongs to this column
            region_for_col = None
            for i, bottom_region in enumerate(bottom_row_regions):
                if bottom_region_cols[i] == col_idx:
                    region_for_col = bottom_region
                    break

            if region_for_col and row_idx < region_for_col['height']:
                # Add this row from the region
                result_row.extend(region_for_col['data'][row_idx])
                # Pad to column width
                padding = col_widths[col_idx] - region_for_col['width']
                result_row.extend([2] * padding)
            else:
                # Pad entire column
                result_row.extend([2] * col_widths[col_idx])
        result.append(result_row)

    return result
