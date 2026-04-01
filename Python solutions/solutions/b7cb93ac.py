def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with scattered colored regions
    2. Output is a 3×4 grid that overlays these regions
    3. Non-8 components are extracted and placed on canvas
    4. Value 8 fills remaining empty cells after overlay
    5. Largest component is placed at (0,0), others positioned relative to it

    Procedure:
    1. Find all non-8 colored components using BFS
    2. Extract bounding box patterns for each component
    3. Sort by area (descending), then by first appearance row
    4. Place largest component at (0,0)
    5. Place subsequent components with column offset scaled from input
    6. For same-color components, first at row 0, rest use scaled row position
    7. Overlay with "first non-zero wins" rule
    8. Fill remaining zeros with 8
    """
    from collections import deque

    rows = len(grid)
    cols = len(grid[0])

    # Find all non-8 colored components
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 8:
                colors.add(grid[r][c])

    components = []

    for color in colors:
        visited = [[False] * cols for _ in range(rows)]

        for start_r in range(rows):
            for start_c in range(cols):
                if grid[start_r][start_c] == color and not visited[start_r][start_c]:
                    # BFS to find connected component
                    cells = []
                    queue = deque([(start_r, start_c)])
                    visited[start_r][start_c] = True

                    while queue:
                        r, c = queue.popleft()
                        cells.append((r, c))

                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == color:
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                    # Extract bounding box
                    min_r = min(r for r, c in cells)
                    max_r = max(r for r, c in cells)
                    min_c = min(c for r, c in cells)
                    max_c = max(c for r, c in cells)

                    bbox_height = max_r - min_r + 1
                    bbox_width = max_c - min_c + 1
                    pattern = [[0] * bbox_width for _ in range(bbox_height)]

                    for r, c in cells:
                        pattern[r - min_r][c - min_c] = color

                    components.append({
                        'color': color,
                        'start_row': min_r,
                        'start_col': min_c,
                        'pattern': pattern,
                        'area': len(cells)
                    })

    # Sort components by start row (ascending), then by area (descending)
    components.sort(key=lambda x: (x['start_row'], -x['area']))

    output_height = 3
    output_width = 4
    output = [[0] * output_width for _ in range(output_height)]

    if not components:
        # Fill with 8 if no components
        for r in range(output_height):
            for c in range(output_width):
                output[r][c] = 8
        return output

    # Track which colors have been placed
    colors_seen = set()

    # Base component (largest) placed at origin
    base_comp = components[0]
    base_col = base_comp['start_col']
    # Check if base pattern has leading zeros at (0,0)
    base_has_leading_zero = base_comp['pattern'][0][0] == 0

    for idx, comp in enumerate(components):
        pattern = comp['pattern']
        pat_h = len(pattern)
        pat_w = len(pattern[0])

        # Handle vertical-to-horizontal rotation for thin vertical patterns
        if pat_h > pat_w and pat_h >= 2 and pat_w == 1:
            # Check if it's a vertical line (all non-zero)
            is_vertical_line = all(pattern[i][0] != 0 for i in range(pat_h))
            if is_vertical_line:
                # Rotate to horizontal
                pattern = [[pattern[i][0] for i in range(pat_h)]]
                pat_h, pat_w = 1, pat_h

        # Determine row placement
        if comp['color'] not in colors_seen:
            # First component of this color goes to row 0
            out_row = 0
            colors_seen.add(comp['color'])
        else:
            # Subsequent components of same color use scaled row
            out_row = (comp['start_row'] * output_height) // rows

        # Determine column placement
        if idx == 0:
            # Base component at column 0
            out_col = 0
        else:
            # Calculate offset from base and scale with rounding
            col_offset = comp['start_col'] - base_col
            scaled_offset = (col_offset * output_width + cols // 2) // cols
            # If base has leading zero and offset is small, place at column 0 to fill gaps
            if base_has_leading_zero and scaled_offset <= 1:
                out_col = 0
            else:
                out_col = max(0, scaled_offset)

        # Place pattern on output (overlay rule: first non-zero wins)
        for dr in range(pat_h):
            for dc in range(pat_w):
                out_r = out_row + dr
                out_c = out_col + dc

                if 0 <= out_r < output_height and 0 <= out_c < output_width:
                    if output[out_r][out_c] == 0 and pattern[dr][dc] != 0:
                        output[out_r][out_c] = pattern[dr][dc]

    # Fill remaining zeros with 8
    for r in range(output_height):
        for c in range(output_width):
            if output[r][c] == 0:
                output[r][c] = 8

    return output
