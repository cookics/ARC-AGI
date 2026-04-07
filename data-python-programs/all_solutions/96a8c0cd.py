def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Single 2 is seed; groups of 1s/3s are objects
    2. Path creates vertical spine that shifts before object rows
    3. In upper part of grid: spine moves to rightmost adjacent position
    4. In lower part: spine moves to leftmost adjacent position
    5. In middle: prefers gaps between groups

    Procedure:
    1. Find seed and all object rows
    2. For each object row, compute target column based on grid position
    3. Draw path: horizontal shifts one row before object rows, vertical otherwise
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find seed
    seed_r, seed_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                seed_r, seed_c = r, c
                break
        if seed_r is not None:
            break

    if seed_r is None:
        return result

    def find_row_groups(r):
        """Find horizontal groups in row r"""
        groups = []
        c = 0
        while c < cols:
            if grid[r][c] in [1, 3]:
                start = c
                val = grid[r][c]
                while c < cols and grid[r][c] == val:
                    c += 1
                groups.append((start, c - 1))
            else:
                c += 1
        return groups

    def get_adjacent_positions(groups):
        """Get all valid adjacent positions to groups"""
        positions = []
        leftmost = min(g[0] for g in groups)
        rightmost = max(g[1] for g in groups)

        # Left edge
        if leftmost > 0:
            positions.append(leftmost - 1)

        # Gaps between groups
        for i in range(len(groups) - 1):
            gap_start = groups[i][1] + 1
            gap_end = groups[i + 1][0] - 1
            for c in range(gap_start, gap_end + 1):
                positions.append(c)

        # Right edge
        if rightmost < cols - 1:
            positions.append(rightmost + 1)

        return positions, leftmost, rightmost

    def determine_target(current_col, groups, row_idx):
        """Determine target column for object row"""
        if not groups:
            return current_col

        positions, leftmost, rightmost = get_adjacent_positions(groups)

        if not positions:
            return current_col

        # Check if current column is already valid
        collides = any(left <= current_col <= right for left, right in groups)
        if not collides and current_col in positions:
            return current_col

        # Categorize positions
        gaps = [c for c in positions if leftmost < c < rightmost]
        left_edge = leftmost - 1 if leftmost > 0 else None
        right_edge = rightmost + 1 if rightmost < cols - 1 else None

        # Decision based on grid position
        # Divide at midpoint: upper half prefers right, lower half prefers left
        if row_idx <= rows / 2:
            # Upper half: prefer rightmost
            if right_edge is not None and right_edge in positions:
                return right_edge
            if gaps:
                return max(gaps)
            if left_edge is not None:
                return left_edge
            return max(positions)
        else:
            # Lower half: prefer leftmost
            if left_edge is not None and left_edge in positions:
                return left_edge
            if gaps:
                return min(gaps)
            if right_edge is not None:
                return right_edge
            return min(positions)

    # Find all object rows
    object_rows = {}
    for r in range(rows):
        groups = find_row_groups(r)
        if groups:
            object_rows[r] = groups

    # Process downward
    current_col = seed_c
    for r in range(seed_r, rows):
        next_r = r + 1
        if next_r in object_rows:
            # Shift before object row
            target_col = determine_target(current_col, object_rows[next_r], next_r)
            for c in range(min(current_col, target_col), max(current_col, target_col) + 1):
                if result[r][c] == 0:
                    result[r][c] = 2
            current_col = target_col
        else:
            # Vertical continuation
            if result[r][current_col] == 0:
                result[r][current_col] = 2

    # Process upward
    current_col = seed_c
    for r in range(seed_r - 1, -1, -1):
        prev_r = r - 1
        if prev_r >= 0 and prev_r in object_rows:
            # Shift before object row
            target_col = determine_target(current_col, object_rows[prev_r], prev_r)
            for c in range(min(current_col, target_col), max(current_col, target_col) + 1):
                if result[r][c] == 0:
                    result[r][c] = 2
            current_col = target_col
        else:
            # Vertical continuation
            if result[r][current_col] == 0:
                result[r][current_col] = 2

    return result
