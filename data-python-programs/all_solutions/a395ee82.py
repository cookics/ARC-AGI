def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a main pattern (largest connected component) and scattered marker cells
    2. For each marker row, place horizontal copies equal to number of markers in that row
    3. Pattern color is inverted based on marker color
    4. Patterns are arranged vertically by marker row order

    Procedure:
    1. Find background color (most common)
    2. Find largest connected component (main pattern)
    3. Extract pattern shape and bounding box
    4. Group markers by row
    5. For each marker row, place copies horizontally based on markers in that row
    """
    from collections import Counter, deque, defaultdict

    rows, cols = len(grid), len(grid[0])

    # Find background color (most common)
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    bg_color = Counter(all_values).most_common(1)[0][0]

    # Find all non-background cells
    non_bg_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != bg_color]

    if not non_bg_cells:
        return grid

    # Find connected components using BFS
    visited = set()
    components = []

    for start_r, start_c in non_bg_cells:
        if (start_r, start_c) in visited:
            continue

        component = []
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] != bg_color:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        components.append(component)

    # Largest component is the pattern
    pattern_component = max(components, key=len)
    pattern_cells = set(pattern_component)

    # Get pattern bounding box
    pattern_rows = [r for r, c in pattern_component]
    pattern_cols = [c for r, c in pattern_component]
    p_r1, p_r2 = min(pattern_rows), max(pattern_rows)
    p_c1, p_c2 = min(pattern_cols), max(pattern_cols)
    p_height = p_r2 - p_r1 + 1
    p_width = p_c2 - p_c1 + 1
    p_center_col = (p_c1 + p_c2) // 2

    # Extract pattern shape (relative coordinates)
    pattern_shape = []
    pattern_color = grid[pattern_component[0][0]][pattern_component[0][1]]
    for r, c in pattern_component:
        pattern_shape.append((r - p_r1, c - p_c1))

    # Find marker cells (non-pattern, non-background)
    markers = [(r, c, grid[r][c]) for r, c in non_bg_cells if (r, c) not in pattern_cells]

    if not markers:
        return grid

    # Group markers by row
    markers_by_row = defaultdict(list)
    for r, c, color in markers:
        markers_by_row[r].append((c, color))

    # Sort each row's markers by column
    for r in markers_by_row:
        markers_by_row[r].sort()

    # Get sorted marker rows
    marker_row_list = sorted(markers_by_row.keys())

    # Find other color (non-pattern color)
    marker_colors = set(color for _, _, color in markers)
    other_color = None
    for color in marker_colors:
        if color != pattern_color:
            other_color = color
            break

    # Create output grid
    result = [[bg_color for _ in range(cols)] for _ in range(rows)]

    # Starting row for output
    out_start_row = p_r1 - p_height

    # Place patterns for each marker row
    for row_idx, marker_row in enumerate(marker_row_list):
        row_markers = markers_by_row[marker_row]
        n_copies = len(row_markers)

        # Calculate vertical position
        base_r = out_start_row + row_idx * p_height

        # Calculate horizontal positions (centered around p_center_col)
        middle_idx = (n_copies - 1) / 2

        for copy_idx, (marker_col, marker_color) in enumerate(row_markers):
            # Determine color for this copy (invert: if marker is pattern color, use other; else use pattern)
            if marker_color == pattern_color:
                copy_color = other_color if other_color else pattern_color
            else:
                copy_color = pattern_color

            # Calculate horizontal position
            if n_copies % 2 == 1:
                # Odd number of copies: middle copy at original position
                middle = n_copies // 2
                offset = copy_idx - middle
                base_c = p_c1 + offset * p_width
            else:
                # Even number of copies: skip the center
                if copy_idx < n_copies // 2:
                    offset = copy_idx - (n_copies // 2)
                else:
                    offset = copy_idx - (n_copies // 2) + 1
                base_c = p_c1 + offset * p_width

            # Place pattern
            for dr, dc in pattern_shape:
                out_r = base_r + dr
                out_c = base_c + dc
                if 0 <= out_r < rows and 0 <= out_c < cols:
                    result[out_r][out_c] = copy_color

    return result
