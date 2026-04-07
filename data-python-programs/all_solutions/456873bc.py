def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The input has regions separated by 3s. The output removes 3s and reflects/copies patterns
    across the separators. Corner positions of rectangular regions get marked with 8s.

    Procedure:
    1. Replace all 3s with 0s
    2. Copy/reflect patterns across regions separated by 3s
    3. Mark corners of rectangular regions with 8s
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find vertical separators (columns of all 3s)
    v_sep_cols = []
    for j in range(cols):
        if all(grid[r][j] == 3 for r in range(rows)):
            v_sep_cols.append(j)

    # Find horizontal separators (rows of all 3s)
    h_sep_rows = []
    for i in range(rows):
        if all(grid[i][c] == 3 for c in range(cols)):
            h_sep_rows.append(i)

    # Replace all 3s with 0s
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 3:
                result[r][c] = 0

    # Handle vertical separation (copy left to separator columns)
    if v_sep_cols:
        left_end = min(v_sep_cols)
        sep_start = min(v_sep_cols)
        sep_width = len(v_sep_cols)

        # For each row, copy pattern from left to separator columns
        for r in range(rows):
            # Copy first sep_width elements from left side to separator columns
            for i in range(sep_width):
                if i < left_end and grid[r][i] == 2:
                    result[r][sep_start + i] = 2

    # Handle horizontal separation (copy top to middle)
    if h_sep_rows:
        top_end = min(h_sep_rows)
        # Copy from top region to separator region
        for r in range(min(h_sep_rows), max(h_sep_rows) + 1):
            for c in range(cols):
                source_row = r - min(h_sep_rows)
                if source_row < top_end and grid[source_row][c] == 2:
                    result[r][c] = 2

    # Mark specific positions with 8s - simplified final approach

    # Step 1: Mark truly isolated 2s (completely alone)
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 2:
                adjacent_2s = 0
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 2:
                        adjacent_2s += 1

                if adjacent_2s == 0:
                    result[r][c] = 8

    # Step 2: Mark corners of the overall bounding box (very selective)
    twos_positions = []
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 2:  # Don't include already marked 8s
                twos_positions.append((r, c))

    if twos_positions:
        min_r = min(pos[0] for pos in twos_positions)
        max_r = max(pos[0] for pos in twos_positions)
        min_c = min(pos[1] for pos in twos_positions)
        max_c = max(pos[1] for pos in twos_positions)

        # Only mark corners of large bounding boxes with strict criteria
        if (max_r - min_r >= 5) and (max_c - min_c >= 5):
            corners = [(min_r, min_c), (min_r, max_c), (max_r, min_c), (max_r, max_c)]

            for cr, cc in corners:
                if result[cr][cc] == 2:
                    # Very strict corner criteria - must have very few neighbors
                    adjacent_2s = 0
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] == 2:
                            adjacent_2s += 1

                    # Only mark true corners with minimal neighbors
                    if adjacent_2s <= 1:
                        result[cr][cc] = 8

    return result
