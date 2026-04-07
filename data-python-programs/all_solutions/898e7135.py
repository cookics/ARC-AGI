def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has scattered marker cells (single isolated cells of one color)
    2. Input has a dominant color region (largest connected component) → becomes background
    3. Output size is 2x the dominant region's bounding box
    4. Other colored shapes are extracted and transformed
    5. Shapes are placed based on their input position mapped to output grid regions

    Procedure:
    1. Find all connected components by color
    2. Identify marker color (has many single-cell components)
    3. Identify background color (largest component) and its bbox
    4. Output size = 2 × dominant_bbox
    5. Extract other shapes and apply transformations (rotation based on aspect ratio)
    6. Map shapes from input regions to output regions
    """
    from collections import defaultdict, deque

    rows, cols = len(grid), len(grid[0])

    # Find all connected components
    def get_components():
        components_by_color = defaultdict(list)
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0 and (r, c) not in visited:
                    color = grid[r][c]
                    component = []
                    queue = deque([(r, c)])
                    visited.add((r, c))

                    while queue:
                        cr, cc = queue.popleft()
                        component.append((cr, cc))

                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if (0 <= nr < rows and 0 <= nc < cols and
                                grid[nr][nc] == color and (nr, nc) not in visited):
                                visited.add((nr, nc))
                                queue.append((nr, nc))

                    components_by_color[color].append(component)

        return components_by_color

    components = get_components()

    # Identify marker color (many single-cell components)
    marker_color = None
    max_singles = 0
    for color, comps in components.items():
        singles = sum(1 for c in comps if len(c) == 1)
        if singles >= 3 and singles > max_singles:
            max_singles = singles
            marker_color = color

    # Find background (largest component)
    background_color = None
    max_size = 0
    dominant_bbox = None

    for color, comps in components.items():
        if color != marker_color:
            for comp in comps:
                if len(comp) > max_size:
                    max_size = len(comp)
                    background_color = color
                    min_r = min(r for r, c in comp)
                    max_r = max(r for r, c in comp)
                    min_c = min(c for r, c in comp)
                    max_c = max(c for r, c in comp)
                    dominant_bbox = (min_r, max_r, min_c, max_c)

    if background_color is None:
        background_color = 0
        dominant_bbox = (0, 0, 0, 0)

    # Calculate output dimensions
    bbox_h = dominant_bbox[1] - dominant_bbox[0] + 1
    bbox_w = dominant_bbox[3] - dominant_bbox[2] + 1
    out_h = bbox_h * 2
    out_w = bbox_w * 2

    # Extract shapes
    shapes = []
    for color, comps in components.items():
        if color != marker_color and color != background_color:
            for comp in comps:
                min_r = min(r for r, c in comp)
                max_r = max(r for r, c in comp)
                min_c = min(c for r, c in comp)
                max_c = max(c for r, c in comp)

                # Extract shape within bbox
                h = max_r - min_r + 1
                w = max_c - min_c + 1
                shape = [[0] * w for _ in range(h)]

                for r, c in comp:
                    shape[r - min_r][c - min_c] = color

                # Transform based on aspect ratio
                if h > w:  # Tall shape: transpose then rotate 180°
                    # Transpose
                    transposed = [[shape[r][c] for r in range(h)] for c in range(w)]
                    th, tw = len(transposed), len(transposed[0])
                    # Rotate 180°
                    final = [[transposed[th - 1 - r][tw - 1 - c] for c in range(tw)] for r in range(th)]
                else:  # Wide or square: just rotate 180°
                    final = [[shape[h - 1 - r][w - 1 - c] for c in range(w)] for r in range(h)]

                shapes.append({
                    'color': color,
                    'shape': final,
                    'pos': (min_r, min_c),
                    'h': len(final),
                    'w': len(final[0])
                })

    # Sort shapes by position (row, then column)
    shapes.sort(key=lambda s: (s['pos'][0], s['pos'][1]))

    # Create output grid
    result = [[background_color] * out_w for _ in range(out_h)]

    # Place shapes in a 2-column grid layout
    padding = 2
    row_spacing = 2

    # For 2-column layout, calculate positions dynamically
    # First column at padding, second column positioned to leave equal margins

    # Group shapes into rows (2 per row)
    shape_rows = []
    i = 0
    while i < len(shapes):
        row_shapes = shapes[i:min(i+2, len(shapes))]
        shape_rows.append(row_shapes)
        i += 2

    # Calculate column positions based on shape widths
    # Find max width for each column
    max_w1 = 0
    max_w2 = 0
    for row_shapes in shape_rows:
        if len(row_shapes) >= 1:
            max_w1 = max(max_w1, row_shapes[0]['w'])
        if len(row_shapes) >= 2:
            max_w2 = max(max_w2, row_shapes[1]['w'])

    # Calculate spacing to center shapes
    # Total available width: out_w
    # We want: padding + max_w1 + spacing + max_w2 + padding = out_w
    # So spacing = out_w - 2*padding - max_w1 - max_w2
    center_spacing = max(2, out_w - 2*padding - max_w1 - max_w2)

    col_positions = [
        padding,
        padding + max_w1 + center_spacing
    ]

    # Place shapes
    current_row = padding
    for row_shapes in shape_rows:
        row_height = max(s['h'] for s in row_shapes)

        for col_idx, shape_info in enumerate(row_shapes):
            shape = shape_info['shape']
            sh, sw = shape_info['h'], shape_info['w']
            start_c = col_positions[col_idx]

            # Place shape
            for r in range(sh):
                for c in range(sw):
                    if shape[r][c] != 0:
                        out_r = current_row + r
                        out_c = start_c + c
                        if 0 <= out_r < out_h and 0 <= out_c < out_w:
                            result[out_r][out_c] = shape[r][c]

        current_row += row_height + row_spacing

    return result
