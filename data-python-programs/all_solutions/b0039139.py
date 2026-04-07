def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is divided by separators (rows or columns of all 1s)
    2. There are 4 sections: 2 pattern sections (0s + other values) and 2 uniform sections (all same value)
    3. Pattern sections define the structure, uniform sections provide colors
    4. Pattern 1 determines output dimensions (height for vertical separators, width for horizontal)
    5. Connected components in pattern 2 determine number of repetitions
    6. Output is pattern 1 repeated with separator rows/columns between repetitions

    Procedure:
    1. Detect separator direction (horizontal or vertical)
    2. Split grid into sections based on separators
    3. Classify sections as pattern (mixed) or uniform (solid color)
    4. Extract bounding boxes of pattern sections
    5. Count connected components in pattern 2
    6. Build output by repeating pattern 1 with separators
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])

    # Detect separators
    h_seps = [r for r in range(rows) if all(grid[r][c] == 1 for c in range(cols))]
    v_seps = [c for c in range(cols) if all(grid[r][c] == 1 for r in range(rows))]

    # Determine separator direction
    is_horizontal = len(h_seps) > 0

    # Extract sections
    def extract_sections(is_horizontal):
        sections = []
        if is_horizontal:
            seps = h_seps + [rows]
            prev = -1
            for sep in seps:
                if prev + 1 < sep:
                    sections.append([grid[r][:] for r in range(prev + 1, sep)])
                prev = sep
        else:
            seps = v_seps + [cols]
            prev = -1
            for sep in seps:
                if prev + 1 < sep:
                    sections.append([[grid[r][c] for c in range(prev + 1, sep)] for r in range(rows)])
                prev = sep
        return sections

    sections = extract_sections(is_horizontal)

    # Classify sections
    def is_uniform(section):
        if not section or not section[0]:
            return False
        first_val = section[0][0]
        if first_val == 0:
            return False
        return all(section[r][c] == first_val for r in range(len(section)) for c in range(len(section[0])))

    patterns = []
    uniforms = []
    for sec in sections:
        if is_uniform(sec):
            uniforms.append(sec)
        else:
            patterns.append(sec)

    # Extract bounding boxes
    def extract_bbox(pattern):
        h, w = len(pattern), len(pattern[0]) if pattern else 0
        min_r, max_r = h, -1
        min_c, max_c = w, -1

        for r in range(h):
            for c in range(w):
                if pattern[r][c] != 0:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        if max_r == -1:
            return []

        bbox = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(pattern[r][c])
            bbox.append(row)
        return bbox

    pattern1 = extract_bbox(patterns[0]) if patterns else []
    pattern2 = extract_bbox(patterns[1]) if len(patterns) > 1 else []

    # Get uniform colors
    color1 = uniforms[0][0][0] if uniforms else 0
    color2 = uniforms[1][0][0] if len(uniforms) > 1 else 0

    # Count connected components in pattern 2
    def count_components(pattern):
        if not pattern:
            return 0

        h, w = len(pattern), len(pattern[0])
        visited = set()
        count = 0

        def bfs(start_r, start_c):
            queue = deque([(start_r, start_c)])
            visited.add((start_r, start_c))

            while queue:
                r, c = queue.popleft()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and pattern[nr][nc] != 0:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        for r in range(h):
            for c in range(w):
                if pattern[r][c] != 0 and (r, c) not in visited:
                    bfs(r, c)
                    count += 1

        return count

    num_repetitions = count_components(pattern2)

    # Build output
    def build_output():
        if not pattern1:
            return [[]]

        p1_h, p1_w = len(pattern1), len(pattern1[0])

        if is_horizontal:
            # Output width = pattern1 width
            # Output height = pattern1 height * repetitions + (repetitions - 1) separators
            out_w = p1_w
            out_h = p1_h * num_repetitions + (num_repetitions - 1)

            result = []
            for rep in range(num_repetitions):
                # Add pattern1
                for r in range(p1_h):
                    row = []
                    for c in range(p1_w):
                        if pattern1[r][c] != 0:
                            row.append(color1)
                        else:
                            row.append(color2)
                    result.append(row)

                # Add separator row (if not last repetition)
                if rep < num_repetitions - 1:
                    result.append([color2] * out_w)
        else:
            # Output height = pattern1 height
            # Output width = pattern1 width * repetitions + (repetitions - 1) separators
            out_h = p1_h
            out_w = p1_w * num_repetitions + (num_repetitions - 1)

            result = [[0] * out_w for _ in range(out_h)]

            col_offset = 0
            for rep in range(num_repetitions):
                # Add pattern1
                for r in range(p1_h):
                    for c in range(p1_w):
                        if pattern1[r][c] != 0:
                            result[r][col_offset + c] = color1
                        else:
                            result[r][col_offset + c] = color2

                col_offset += p1_w

                # Add separator column (if not last repetition)
                if rep < num_repetitions - 1:
                    for r in range(out_h):
                        result[r][col_offset] = color2
                    col_offset += 1

        return result

    return build_output()
