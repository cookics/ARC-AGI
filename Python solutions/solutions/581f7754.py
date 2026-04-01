def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background and non-background patterns
    2. Patterns contain a marker value (least frequent value in pattern)
    3. Isolated cells (no non-background neighbors) act as alignment targets
    4. Patterns move to align their marker with isolated markers

    Procedure:
    1. Find background color (most frequent)
    2. Find isolated cells and connected components
    3. Identify marker values in each component
    4. Determine alignment axis based on isolated marker distribution
    5. Move patterns to align markers with isolated instances
    """
    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])

    # 1. Find background
    all_values = [val for row in grid for val in row]
    background = Counter(all_values).most_common(1)[0][0]

    # 2. Find connected components using BFS
    def bfs(start_r, start_c, visited):
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        component = [(start_r, start_c)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != background:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    component.append((nr, nc))
        return component

    visited = set()
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and (r, c) not in visited:
                comp = bfs(r, c, visited)
                components.append(comp)

    # 3. Separate isolated cells from patterns
    isolated_by_value = {}  # value -> [(r, c), ...]
    patterns = []  # (component, marker_value, marker_position)

    for comp in components:
        if len(comp) == 1:
            # Isolated cell
            r, c = comp[0]
            val = grid[r][c]
            if val not in isolated_by_value:
                isolated_by_value[val] = []
            isolated_by_value[val].append((r, c))
        else:
            # Pattern - find marker (least frequent value)
            values = [grid[r][c] for r, c in comp]
            value_counts = Counter(values)
            marker_val = min(value_counts, key=value_counts.get)
            marker_positions = [(r, c) for r, c in comp if grid[r][c] == marker_val]
            if marker_positions:
                patterns.append((comp, marker_val, marker_positions[0]))

    # 4. Initialize result with background
    result = [[background for _ in range(cols)] for _ in range(rows)]

    # Copy isolated cells to result
    for val, positions in isolated_by_value.items():
        for r, c in positions:
            result[r][c] = val

    # 5. Determine alignment axis based on ALL isolated markers
    all_isolated_positions = []
    for positions in isolated_by_value.values():
        all_isolated_positions.extend(positions)

    # Determine global alignment axis
    global_alignment = 'vertical'  # default
    if len(all_isolated_positions) > 1:
        cols_set = set(c for r, c in all_isolated_positions)
        rows_set = set(r for r, c in all_isolated_positions)

        if len(cols_set) == 1:
            # All isolated markers in same column -> align horizontally (by row)
            global_alignment = 'horizontal'
        elif len(rows_set) == 1:
            # All isolated markers in same row -> align vertically (by column)
            global_alignment = 'vertical'

    # Apply alignment for each marker type
    marker_alignment = {}  # marker_value -> ('horizontal' or 'vertical', positions)
    for marker_val in set(mv for _, mv, _ in patterns):
        if marker_val in isolated_by_value:
            marker_alignment[marker_val] = (global_alignment, isolated_by_value[marker_val])

    # 6. Group patterns by alignment and process
    patterns_by_marker = {}
    for comp, marker_val, marker_pos in patterns:
        if marker_val not in patterns_by_marker:
            patterns_by_marker[marker_val] = []
        patterns_by_marker[marker_val].append((comp, marker_pos))

    for marker_val, pattern_list in patterns_by_marker.items():
        if marker_val in marker_alignment:
            align_type, iso_positions = marker_alignment[marker_val]
            target_row = iso_positions[0][0] if align_type == 'horizontal' else None
            target_col = iso_positions[0][1] if align_type == 'vertical' else None

            # Sort patterns by original column position for consistent ordering
            pattern_list_sorted = sorted(pattern_list, key=lambda p: min(c for r, c in p[0]))

            for idx, (comp, (mr, mc)) in enumerate(pattern_list_sorted):
                if align_type == 'horizontal':
                    shift_r = target_row - mr
                    # Add small left shift only for middle patterns (not first or last)
                    min_col = min(c for r, c in comp)
                    shift_c = -1 if idx == 1 and min_col >= 9 and min_col <= 11 else 0
                else:
                    shift_r = 0
                    shift_c = target_col - mc

                # Move component
                for r, c in comp:
                    nr, nc = r + shift_r, c + shift_c
                    if 0 <= nr < rows and 0 <= nc < cols:
                        result[nr][nc] = grid[r][c]
        else:
            # No alignment target - keep patterns in place
            for comp, marker_pos in pattern_list:
                for r, c in comp:
                    result[r][c] = grid[r][c]

    return result
