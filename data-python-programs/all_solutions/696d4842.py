def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. L-shapes extend toward isolated markers
    2. When extending vertically N cells, recolor N cells at opposite horizontal end
    3. When extending horizontally N cells, recolor N cells at opposite vertical end
    4. Recoloring starts from the edge (leftmost for horizontal, topmost for vertical)

    Procedure:
    1. BFS to find all connected components
    2. Separate single-cell markers from multi-cell shapes
    3. For each shape, find aligned marker outside its bounds
    4. Extend along aligned axis toward marker
    5. Recolor perpendicular segment at opposite end
    """
    from collections import deque

    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    visited = set()

    def bfs(sr, sc):
        color = grid[sr][sc]
        cells = []
        q = deque([(sr, sc)])
        visited.add((sr, sc))

        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and
                    (nr, nc) not in visited and grid[nr][nc] == color):
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return color, cells

    components = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] != 0 and (i, j) not in visited:
                components.append(bfs(i, j))

    markers = [(color, cells[0]) for color, cells in components if len(cells) == 1]
    shapes = [(color, cells) for color, cells in components if len(cells) > 1]

    used_markers = set()

    for shape_color, shape_cells in shapes:
        # Group shape cells by row and column
        by_row = {}
        by_col = {}
        for r, c in shape_cells:
            by_row.setdefault(r, []).append(c)
            by_col.setdefault(c, []).append(r)

        # Find matching marker
        matching_marker = None
        marker_idx = None

        for idx, (marker_color, (m_r, m_c)) in enumerate(markers):
            if idx in used_markers or marker_color == shape_color:
                continue

            # Check if marker aligns with a column in shape and is outside
            if m_c in by_col:
                rows_in_col = by_col[m_c]
                if m_r < min(rows_in_col) or m_r > max(rows_in_col):
                    matching_marker = (marker_color, m_r, m_c)
                    marker_idx = idx
                    break
            # Check if marker aligns with a row in shape and is outside
            elif m_r in by_row:
                cols_in_row = by_row[m_r]
                if m_c < min(cols_in_row) or m_c > max(cols_in_row):
                    matching_marker = (marker_color, m_r, m_c)
                    marker_idx = idx
                    break

        if not matching_marker:
            continue

        used_markers.add(marker_idx)
        marker_color, m_r, m_c = matching_marker
        extension_count = 0

        # CASE 1: Vertical extension (marker aligned with column)
        if m_c in by_col:
            rows_in_col = sorted(by_col[m_c])
            min_row_in_col = min(rows_in_col)
            max_row_in_col = max(rows_in_col)

            if m_r < min_row_in_col:
                # Marker above, extend downward
                for r in range(m_r + 1, min_row_in_col):
                    if result[r][m_c] == 0:
                        result[r][m_c] = shape_color
                        extension_count += 1

                # Recolor at BOTTOM horizontal segment
                # Find row with horizontal segment
                rows_with_horizontal = [r for r in by_row if len(by_row[r]) > 1]
                if rows_with_horizontal:
                    # Pick the bottommost row with a horizontal segment
                    target_row = max(rows_with_horizontal)
                    cols_to_recolor = sorted(by_row[target_row])
                    for i in range(min(extension_count, len(cols_to_recolor))):
                        result[target_row][cols_to_recolor[i]] = marker_color

            elif m_r > max_row_in_col:
                # Marker below, extend upward
                for r in range(max_row_in_col + 1, m_r):
                    if result[r][m_c] == 0:
                        result[r][m_c] = shape_color
                        extension_count += 1

                # Recolor at TOP horizontal segment
                rows_with_horizontal = [r for r in by_row if len(by_row[r]) > 1]
                if rows_with_horizontal:
                    # Pick the topmost row with a horizontal segment
                    target_row = min(rows_with_horizontal)
                    cols_to_recolor = sorted(by_row[target_row])
                    for i in range(min(extension_count, len(cols_to_recolor))):
                        result[target_row][cols_to_recolor[i]] = marker_color

        # CASE 2: Horizontal extension (marker aligned with row)
        elif m_r in by_row:
            cols_in_row = sorted(by_row[m_r])
            min_col_in_row = min(cols_in_row)
            max_col_in_row = max(cols_in_row)

            if m_c < min_col_in_row:
                # Marker to left, extend rightward
                for c in range(m_c + 1, min_col_in_row):
                    if result[m_r][c] == 0:
                        result[m_r][c] = shape_color
                        extension_count += 1

                # Recolor at RIGHTMOST vertical segment
                cols_with_vertical = [c for c in by_col if len(by_col[c]) > 1]
                if cols_with_vertical:
                    # Pick the rightmost column with a vertical segment
                    target_col = max(cols_with_vertical)
                    rows_to_recolor = sorted(by_col[target_col])
                    for i in range(min(extension_count, len(rows_to_recolor))):
                        result[rows_to_recolor[i]][target_col] = marker_color

            elif m_c > max_col_in_row:
                # Marker to right, extend leftward
                for c in range(max_col_in_row + 1, m_c):
                    if result[m_r][c] == 0:
                        result[m_r][c] = shape_color
                        extension_count += 1

                # Recolor at LEFTMOST vertical segment
                cols_with_vertical = [c for c in by_col if len(by_col[c]) > 1]
                if cols_with_vertical:
                    # Pick the leftmost column with a vertical segment
                    target_col = min(cols_with_vertical)
                    rows_to_recolor = sorted(by_col[target_col])
                    for i in range(min(extension_count, len(rows_to_recolor))):
                        result[rows_to_recolor[i]][target_col] = marker_color

    return result
