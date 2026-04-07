def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a pattern sequence (non-zero, non-5 values) and regions of 5s forming diagonal staircases
    2. Pattern is applied to each row of 5s with rotation and repetition rules
    3. For components with single diagonal (no horizontal shift): pattern rotates backward each row, fills with first value
    4. For components with horizontal shift: pattern applies progressively (1,2,3,... values) then rotates when shift occurs

    Procedure:
    1. Extract pattern sequence from non-zero, non-5 values
    2. Find all connected components of 5s
    3. For each component, determine if it has horizontal shift or is pure diagonal
    4. Apply appropriate filling rule based on component type
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Extract pattern
    sequence = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in [0, 5]:
                sequence.append(grid[r][c])

    if not sequence:
        return result

    # Find connected components of 5s
    visited = [[False] * cols for _ in range(rows)]

    def dfs(r, c, component):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 5:
            return
        visited[r][c] = True
        component.append((r, c))
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            dfs(r+dr, c+dc, component)

    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5 and not visited[r][c]:
                comp = []
                dfs(r, c, comp)
                if comp:
                    components.append(comp)

    # Process each component
    for comp in components:
        # Group by row
        rows_dict = {}
        for cr, cc in comp:
            if cr not in rows_dict:
                rows_dict[cr] = []
            rows_dict[cr].append(cc)

        sorted_rows = sorted(rows_dict.keys())
        for r in sorted_rows:
            rows_dict[r].sort()

        # Check if component has horizontal shift
        left_edges = [rows_dict[r][0] for r in sorted_rows]
        has_horizontal_shift = len(set(left_edges)) > 1

        # Process each row
        for ridx, row in enumerate(sorted_rows):
            cols_list = rows_dict[row]
            width = len(cols_list)

            if not has_horizontal_shift:
                # Pure diagonal: rotate backward, fill with first
                # Detect narrowing
                widths_list = [len(rows_dict[r]) for r in sorted_rows]
                max_width = max(widths_list)
                max_width_idx = widths_list.index(max_width)
                is_narrowing = ridx > max_width_idx and width < max_width

                if is_narrowing:
                    # During narrowing, keep same starting index as max width row
                    start_idx = max_width_idx % len(sequence)
                    num_vals = max(1, width - len(sequence) + 1)
                else:
                    # Growing: rotate starting index
                    start_idx = ridx % len(sequence)
                    num_vals = min(ridx + 1, len(sequence))

                # Build values list going backward from start_idx
                vals = []
                for i in range(num_vals):
                    vals.append(sequence[(start_idx - i) % len(sequence)])

                # Fill row: first value repeats, then other values
                for i, c in enumerate(cols_list):
                    if i < width - num_vals + 1:
                        result[row][c] = vals[0]
                    else:
                        result[row][c] = vals[i - (width - num_vals + 1) + 1]
            else:
                # Has horizontal shift: progressive then rotate
                shift = left_edges[ridx] - left_edges[0]

                # Find max width and detect narrowing
                widths_list = [len(rows_dict[r]) for r in sorted_rows]
                max_width = max(widths_list)
                max_width_idx = widths_list.index(max_width)
                is_narrowing = ridx > max_width_idx and width < max_width

                if shift == 0:
                    # No shift yet: progressive
                    num_vals = min(ridx + 1, len(sequence))
                    start_idx = 0
                elif is_narrowing:
                    # Narrowing: use fewer values
                    num_vals = max(1, width - len(sequence) + 1)
                    start_idx = shift % len(sequence)
                else:
                    # Shifted but growing/stable: use full pattern
                    num_vals = len(sequence)
                    start_idx = shift % len(sequence)

                # Fill row
                for i, c in enumerate(cols_list):
                    if i < num_vals:
                        result[row][c] = sequence[(start_idx + i) % len(sequence)]
                    else:
                        result[row][c] = sequence[(start_idx + num_vals - 1) % len(sequence)]

    return result
