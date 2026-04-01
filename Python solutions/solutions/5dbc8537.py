def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input divided by separator (uniform row/column) into template and objects sections
    2. Template has structure color (preserved) and placeholder color (regions to fill)
    3. Objects section has background + various colored objects
    4. Find connected components of placeholders in template
    5. Extract objects (by color) from objects section
    6. Match placeholders to objects by scanning order, fill placeholders with object colors

    Procedure:
    1. Find separator, split into template and objects
    2. Identify template's structure and placeholder colors
    3. Find placeholder regions (connected components)
    4. Extract object colors from objects section
    5. Sort placeholders and objects by position
    6. Fill each placeholder with corresponding object color
    """

    from collections import Counter, deque

    rows, cols = len(grid), len(grid[0])

    # Find vertical separator
    v_sep = -1
    for c in range(cols):
        if len(set(grid[r][c] for r in range(rows))) == 1:
            if c > 0 and c < cols - 1:
                v_sep = c
                break

    # Find horizontal separator
    h_sep = -1
    for r in range(rows):
        if len(set(grid[r])) == 1:
            if r > 0 and r < rows - 1:
                h_sep = r
                break

    if v_sep >= 0:
        # Vertical split
        sep_val = grid[0][v_sep]
        left = [[grid[r][c] for c in range(v_sep)] for r in range(rows)]
        right = [[grid[r][c] for c in range(v_sep+1, cols)] for r in range(rows)]

        # Determine template (fewer unique values)
        left_cnt = Counter(v for row in left for v in row)
        right_cnt = Counter(v for row in right for v in row)

        template, objects = (left, right) if len(left_cnt) < len(right_cnt) else (right, left)

        # Get template colors
        temp_counts = Counter(v for row in template for v in row).most_common()
        struct_val = temp_counts[0][0]
        placeholder_val = temp_counts[1][0]

        # Get objects background
        obj_bg = Counter(v for row in objects for v in row).most_common()[0][0]

        # Find placeholder regions (connected components)
        th, tw = len(template), len(template[0])
        visited = [[False] * tw for _ in range(th)]
        placeholders = []

        def bfs_placeholder(sr, sc):
            cells = []
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < th and 0 <= nc < tw and not visited[nr][nc]:
                        if template[nr][nc] == placeholder_val:
                            visited[nr][nc] = True
                            q.append((nr, nc))
            return cells

        for r in range(th):
            for c in range(tw):
                if template[r][c] == placeholder_val and not visited[r][c]:
                    region = bfs_placeholder(r, c)
                    min_r = min(rr for rr, cc in region)
                    min_c = min(cc for rr, cc in region)
                    placeholders.append((min_r, min_c, region))

        placeholders.sort(key=lambda x: (x[0], x[1]))

        # Extract object regions (connected components) from objects section
        oh, ow = len(objects), len(objects[0])
        obj_visited = [[False] * ow for _ in range(oh)]
        object_regions = []

        def bfs_object(sr, sc, color):
            cells = []
            q = deque([(sr, sc)])
            obj_visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < oh and 0 <= nc < ow and not obj_visited[nr][nc]:
                        if objects[nr][nc] == color:
                            obj_visited[nr][nc] = True
                            q.append((nr, nc))
            return cells

        for r in range(oh):
            for c in range(ow):
                if objects[r][c] != obj_bg and not obj_visited[r][c]:
                    color = objects[r][c]
                    region = bfs_object(r, c, color)
                    min_r = min(rr for rr, cc in region)
                    min_c = min(cc for rr, cc in region)
                    object_regions.append((min_r, min_c, color))

        object_regions.sort(key=lambda x: (x[0], x[1]))

        # Build result with template + separator column
        result = [[template[r][c] for c in range(tw)] + [sep_val] for r in range(th)]

        for i, (_, _, region) in enumerate(placeholders):
            if i < len(object_regions):
                fill_color = object_regions[i][2]
                for r, c in region:
                    result[r][c] = fill_color

        return result

    elif h_sep >= 0:
        # Horizontal split
        top = [[grid[r][c] for c in range(cols)] for r in range(h_sep)]
        bottom = [[grid[r][c] for c in range(cols)] for r in range(h_sep+1, rows)]

        # Determine template
        top_cnt = Counter(v for row in top for v in row)
        bottom_cnt = Counter(v for row in bottom for v in row)

        template, objects = (bottom, top) if len(bottom_cnt) < len(top_cnt) else (top, bottom)

        # Get template colors
        temp_counts = Counter(v for row in template for v in row).most_common()
        struct_val = temp_counts[0][0]
        placeholder_val = temp_counts[1][0]

        # Get objects background
        obj_bg = Counter(v for row in objects for v in row).most_common()[0][0]

        # Find placeholder regions
        th, tw = len(template), len(template[0])
        visited = [[False] * tw for _ in range(th)]
        placeholders = []

        def bfs_placeholder(sr, sc):
            cells = []
            q = deque([(sr, sc)])
            visited[sr][sc] = True
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < th and 0 <= nc < tw and not visited[nr][nc]:
                        if template[nr][nc] == placeholder_val:
                            visited[nr][nc] = True
                            q.append((nr, nc))
            return cells

        for r in range(th):
            for c in range(tw):
                if template[r][c] == placeholder_val and not visited[r][c]:
                    region = bfs_placeholder(r, c)
                    min_r = min(rr for rr, cc in region)
                    min_c = min(cc for rr, cc in region)
                    placeholders.append((min_r, min_c, region))

        placeholders.sort(key=lambda x: (x[0], x[1]))

        # Extract object colors
        oh, ow = len(objects), len(objects[0])
        object_colors = []
        for color in set(v for row in objects for v in row):
            if color != obj_bg:
                cells = [(r, c) for r in range(oh) for c in range(ow) if objects[r][c] == color]
                if cells:
                    min_r = min(r for r, c in cells)
                    min_c = min(c for r, c in cells)
                    object_colors.append((min_r, min_c, color))

        object_colors.sort(key=lambda x: (x[0], x[1]))

        # Build result
        result = [row[:] for row in template]

        for i, (_, _, region) in enumerate(placeholders):
            if i < len(object_colors):
                fill_color = object_colors[i][2]
                for r, c in region:
                    result[r][c] = fill_color

        return result

    return grid
