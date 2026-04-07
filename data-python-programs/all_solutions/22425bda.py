def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with background value 7
    2. Certain non-7 values form complete lines (horizontal, vertical, or diagonal)
    3. Lines intersect showing values from both lines
    4. Output is list of line values in specific order

    Procedure:
    1. For each distinct non-7 value, check if it forms a line by examining its positions
    2. A value forms a line if its positions align in a row, column, or diagonal pattern
    3. Sort lines by position (ascending for ≤4 lines, descending for more)
    4. Merge vertical/horizontal by position, then add diagonals
    5. Return ordered list of line values
    """
    from collections import Counter

    if not grid or not grid[0]:
        return [[]]

    rows = len(grid)
    cols = len(grid[0])
    background = 7

    # Collect positions of each non-background value
    value_positions = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                val = grid[r][c]
                if val not in value_positions:
                    value_positions[val] = []
                value_positions[val].append((r, c))

    lines = []  # (value, type, position)

    # For each value, check if it forms a line
    for val, positions in value_positions.items():
        if len(positions) < 3:
            continue

        # Check if all positions are in the same row
        rows_set = set(r for r, c in positions)
        if len(rows_set) == 1:
            row = list(rows_set)[0]
            # Check if this value fills most of the row
            row_size = sum(1 for c in range(cols) if grid[row][c] != background)
            if len(positions) >= row_size * 0.7:
                lines.append((val, 'horizontal', row))
                continue

        # Check if all positions are in the same column
        cols_set = set(c for r, c in positions)
        if len(cols_set) == 1:
            col = list(cols_set)[0]
            col_size = sum(1 for r in range(rows) if grid[r][col] != background)
            if len(positions) >= col_size * 0.7:
                lines.append((val, 'vertical', col))
                continue

        # Check if positions form a main diagonal (row - col = constant)
        diag_offsets = [r - c for r, c in positions]
        offset_counts = Counter(diag_offsets)
        most_common_offset, count = offset_counts.most_common(1)[0]
        if count >= len(positions) * 0.7 and count >= 3:
            lines.append((val, 'main_diagonal', most_common_offset))
            continue

        # Check if positions form an anti-diagonal (row + col = constant)
        diag_sums = [r + c for r, c in positions]
        sum_counts = Counter(diag_sums)
        most_common_sum, count = sum_counts.most_common(1)[0]
        if count >= len(positions) * 0.7 and count >= 3:
            lines.append((val, 'anti_diagonal', most_common_sum))
            continue

    # Determine sort order
    num_lines = len(lines)
    ascending = num_lines <= 4

    # Separate by type
    verticals = [(v, p) for v, t, p in lines if t == 'vertical']
    horizontals = [(v, p) for v, t, p in lines if t == 'horizontal']
    main_diags = [(v, p) for v, t, p in lines if t == 'main_diagonal']
    anti_diags = [(v, p) for v, t, p in lines if t == 'anti_diagonal']

    # Sort and merge
    result = []

    if ascending:
        verticals.sort(key=lambda x: x[1])
        horizontals.sort(key=lambda x: x[1])

        # Find where each diagonal should be inserted
        diag_insertion_pos = {}
        for diag_val, diag_offset in main_diags + anti_diags:
            # Find the highest position among lines this diagonal intersects
            max_pos = -1
            # Check verticals
            for v_val, v_pos in verticals:
                if v_val == diag_val:
                    continue
                # Check if diagonal intersects this vertical
                for r in range(rows):
                    if grid[r][v_pos] == diag_val:
                        max_pos = max(max_pos, v_pos)
                        break
            # Check horizontals
            for h_val, h_pos in horizontals:
                if h_val == diag_val:
                    continue
                for c in range(cols):
                    if grid[h_pos][c] == diag_val:
                        max_pos = max(max_pos, h_pos)
                        break
            diag_insertion_pos[diag_val] = max_pos

        # Merge by position (vertical before horizontal at same position)
        v_idx, h_idx = 0, 0
        diag_values = [v for v, _ in main_diags + anti_diags]
        inserted_diags = set()

        while v_idx < len(verticals) or h_idx < len(horizontals):
            current_pos = None
            if v_idx < len(verticals) and (h_idx >= len(horizontals) or verticals[v_idx][1] < horizontals[h_idx][1]):
                result.append(verticals[v_idx][0])
                current_pos = verticals[v_idx][1]
                v_idx += 1
            elif h_idx < len(horizontals) and (v_idx >= len(verticals) or horizontals[h_idx][1] < verticals[v_idx][1]):
                result.append(horizontals[h_idx][0])
                current_pos = horizontals[h_idx][1]
                h_idx += 1
            else:  # Same position
                result.append(verticals[v_idx][0])
                result.append(horizontals[h_idx][0])
                current_pos = verticals[v_idx][1]
                v_idx += 1
                h_idx += 1

            # Insert diagonals that should come after this position
            for dval in diag_values:
                if dval not in inserted_diags and diag_insertion_pos.get(dval, -1) <= current_pos:
                    # Check if next position is higher
                    next_pos = float('inf')
                    if v_idx < len(verticals):
                        next_pos = min(next_pos, verticals[v_idx][1])
                    if h_idx < len(horizontals):
                        next_pos = min(next_pos, horizontals[h_idx][1])
                    if diag_insertion_pos.get(dval, -1) < next_pos:
                        result.append(dval)
                        inserted_diags.add(dval)

        # Add any remaining diagonals
        for dval in diag_values:
            if dval not in inserted_diags:
                result.append(dval)
    else:
        # Descending order
        verticals.sort(key=lambda x: x[1], reverse=True)
        horizontals.sort(key=lambda x: x[1], reverse=True)

        # Find where each diagonal should be inserted (highest intersection position)
        diag_insertion_pos = {}
        for diag_val, diag_offset in main_diags + anti_diags:
            min_pos = float('inf')
            # Check verticals
            for v_val, v_pos in verticals:
                if v_val == diag_val:
                    continue
                for r in range(rows):
                    if grid[r][v_pos] == diag_val:
                        min_pos = min(min_pos, v_pos)
                        break
            # Check horizontals
            for h_val, h_pos in horizontals:
                if h_val == diag_val:
                    continue
                for c in range(cols):
                    if grid[h_pos][c] == diag_val:
                        min_pos = min(min_pos, h_pos)
                        break
            diag_insertion_pos[diag_val] = min_pos

        v_idx, h_idx = 0, 0
        diag_values = [v for v, _ in main_diags + anti_diags]
        inserted_diags = set()

        while v_idx < len(verticals) or h_idx < len(horizontals):
            current_pos = None
            if v_idx < len(verticals) and (h_idx >= len(horizontals) or verticals[v_idx][1] > horizontals[h_idx][1]):
                result.append(verticals[v_idx][0])
                current_pos = verticals[v_idx][1]
                v_idx += 1
            elif h_idx < len(horizontals) and (v_idx >= len(verticals) or horizontals[h_idx][1] > verticals[v_idx][1]):
                result.append(horizontals[h_idx][0])
                current_pos = horizontals[h_idx][1]
                h_idx += 1
            else:  # Same position
                result.append(verticals[v_idx][0])
                result.append(horizontals[h_idx][0])
                current_pos = verticals[v_idx][1]
                v_idx += 1
                h_idx += 1

            # Insert diagonals
            for dval in diag_values:
                if dval not in inserted_diags and diag_insertion_pos.get(dval, float('inf')) >= current_pos:
                    next_pos = -1
                    if v_idx < len(verticals):
                        next_pos = max(next_pos, verticals[v_idx][1])
                    if h_idx < len(horizontals):
                        next_pos = max(next_pos, horizontals[h_idx][1])
                    if diag_insertion_pos.get(dval, float('inf')) > next_pos:
                        result.append(dval)
                        inserted_diags.add(dval)

        for dval in diag_values:
            if dval not in inserted_diags:
                result.append(dval)

    return [result] if result else [[]]
