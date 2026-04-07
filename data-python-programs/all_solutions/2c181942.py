def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Find 4-row core with 4+ distinct colors
    2. Core cells ALWAYS stay in exact positions
    3. Each core color may have scattered instances outside core
    4. Scattered cells map to positions adjacent to core
    5. Row/column structure of scattered components is preserved

    Procedure:
    1. Find 4-row core with maximum distinct colors
    2. Identify core component for each color
    3. Place core cells in output at exact positions
    4. For scattered components: map to adjacent positions around core
    5. Preserve spatial structure during mapping
    """
    rows, cols = len(grid), len(grid[0])

    # Helper: Find connected components using 8-connectivity
    def find_components(positions):
        visited = set()
        components = []
        pos_set = set(positions)

        for start_r, start_c in positions:
            if (start_r, start_c) in visited:
                continue

            comp = []
            queue = [(start_r, start_c)]
            visited.add((start_r, start_c))

            while queue:
                r, c = queue.pop(0)
                comp.append((r, c))
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in pos_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            components.append(comp)

        return components

    # Collect all positions by color
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                color = grid[r][c]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append((r, c))

    if not color_positions:
        return [[8] * cols for _ in range(rows)]

    # Find 4-row core with maximum distinct colors
    best_core_start = None
    max_colors = 0

    for r_start in range(rows - 3):
        colors_in_window = set()
        for r in range(r_start, r_start + 4):
            for c in range(cols):
                if grid[r][c] != 8:
                    colors_in_window.add(grid[r][c])

        if len(colors_in_window) > max_colors:
            max_colors = len(colors_in_window)
            best_core_start = r_start

    if best_core_start is None or max_colors < 4:
        return [[8] * cols for _ in range(rows)]

    core_start = best_core_start
    core_end = core_start + 4  # exclusive

    # Extract core color positions
    core_colors = {}
    for r in range(core_start, core_end):
        for c in range(cols):
            if grid[r][c] != 8:
                color = grid[r][c]
                if color not in core_colors:
                    core_colors[color] = []
                core_colors[color].append((r, c))

    # Initialize output grid
    result = [[8] * cols for _ in range(rows)]

    # Process each core color
    for color in core_colors:
        all_positions = color_positions[color]
        components = find_components(all_positions)

        # Identify core component vs scattered components
        core_component = None
        scattered_components = []

        for comp in components:
            has_core_cell = any(core_start <= r < core_end for r, c in comp)
            if has_core_cell:
                core_component = comp
            else:
                scattered_components.append(comp)

        if core_component is None:
            continue

        # Always place core cells
        for r, c in core_component:
            result[r][c] = color

        # Process scattered components if they exist
        if not scattered_components:
            continue

        # Get core component bounds and center
        core_rows = [r for r, c in core_component]
        core_cols = [c for r, c in core_component]
        core_r_center = sum(core_rows) / len(core_rows)
        core_c_center = sum(core_cols) / len(core_cols)
        core_c_min = min(core_cols)
        core_c_max = max(core_cols)

        # Process the closest scattered component
        # Distance measured by vertical distance from core center
        def component_distance(comp):
            comp_r_center = sum(r for r, c in comp) / len(comp)
            return abs(comp_r_center - core_r_center)

        closest_scattered = min(scattered_components, key=component_distance)

        # Get scattered component properties
        sc_rows = [r for r, c in closest_scattered]
        sc_cols = [c for r, c in closest_scattered]
        sc_r_min = min(sc_rows)
        sc_r_max = max(sc_rows)
        sc_c_min = min(sc_cols)
        sc_c_max = max(sc_cols)
        sc_r_center = sum(sc_rows) / len(sc_rows)
        sc_c_center = sum(sc_cols) / len(sc_cols)

        # Determine position relative to core
        is_above_core = sc_r_center < core_r_center
        is_left_of_core = sc_c_center < core_c_center

        # Build row-by-row structure of scattered component
        sc_by_row = {}
        for r, c in closest_scattered:
            if r not in sc_by_row:
                sc_by_row[r] = []
            sc_by_row[r].append(c)

        # Map scattered cells to output positions
        # Column mapping: place adjacent to core
        if is_left_of_core:
            # Place to the left of core
            col_offset = core_c_min - sc_c_max - 1
        else:
            # Place to the right of core
            col_offset = core_c_max + 1 - sc_c_min

        # Row mapping depends on position
        sorted_sc_rows = sorted(sc_by_row.keys())

        if is_above_core:
            # Map scattered rows to rows immediately above core
            # The highest scattered row maps to row just above core
            for i, original_r in enumerate(reversed(sorted_sc_rows)):
                target_r = core_start - (len(sorted_sc_rows) - i)
                if target_r < 0:
                    continue
                for original_c in sc_by_row[original_r]:
                    target_c = original_c + col_offset
                    if 0 <= target_c < cols:
                        result[target_r][target_c] = color
        else:
            # Map scattered rows to rows immediately below core
            # The lowest scattered row maps to row just below core
            for i, original_r in enumerate(sorted_sc_rows):
                target_r = core_end + i
                if target_r >= rows:
                    continue
                for original_c in sc_by_row[original_r]:
                    target_c = original_c + col_offset
                    if 0 <= target_c < cols:
                        result[target_r][target_c] = color

    return result
