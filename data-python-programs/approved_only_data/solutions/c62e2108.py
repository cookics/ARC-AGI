def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing lines of 1s (horizontal or vertical) and template patterns made of other colors (2, 3, 4, 8, etc.) in rectangular shapes.
    2. Output is a grid where template patterns extend toward nearby 1-lines and create tiled repetitions.
    3. The 1s themselves disappear in the output grid.
    4. Each template pattern extends in directions indicated by nearby 1-lines.
    5. When extensions cross rows/columns containing 1-lines, those entire rows/columns get filled with repetitions of the template pattern.

    Procedure:
    1. Find all horizontal and vertical 1-lines in the input grid.
    2. Find all template patterns (rectangular regions of non-zero, non-1 values).
    3. For each template pattern, extend it toward nearby 1-lines.
    4. Fill crossing rows/columns with pattern repetitions using tiling logic.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    # Find horizontal 1-lines (consecutive 1s in rows)
    horizontal_1_lines = []
    for r in range(rows):
        start = None
        for c in range(cols):
            if grid[r][c] == 1:
                if start is None:
                    start = c
            else:
                if start is not None:
                    horizontal_1_lines.append((r, start, c - 1))
                    start = None
        if start is not None:
            horizontal_1_lines.append((r, start, cols - 1))

    # Find vertical 1-lines (consecutive 1s in columns)
    vertical_1_lines = []
    for c in range(cols):
        start = None
        for r in range(rows):
            if grid[r][c] == 1:
                if start is None:
                    start = r
            else:
                if start is not None:
                    vertical_1_lines.append((c, start, r - 1))
                    start = None
        if start is not None:
            vertical_1_lines.append((c, start, rows - 1))

    # Find template patterns (connected regions of same non-zero, non-1 values)
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    template_patterns = []

    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0 and grid[r][c] != 1:
                # BFS to find connected component
                pattern_color = grid[r][c]
                pattern_cells = []
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    curr_r, curr_c = queue.pop(0)
                    pattern_cells.append((curr_r, curr_c))

                    # Check 4 directions
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and not visited[nr][nc]
                            and grid[nr][nc] == pattern_color
                        ):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                # Get bounding box
                min_r = min(cell[0] for cell in pattern_cells)
                max_r = max(cell[0] for cell in pattern_cells)
                min_c = min(cell[1] for cell in pattern_cells)
                max_c = max(cell[1] for cell in pattern_cells)

                template_patterns.append(
                    {
                        "color": pattern_color,
                        "cells": pattern_cells,
                        "min_r": min_r,
                        "max_r": max_r,
                        "min_c": min_c,
                        "max_c": max_c,
                    }
                )

    # For each template pattern, extend it toward nearby 1-lines
    for pattern in template_patterns:
        color = pattern["color"]
        min_r, max_r = pattern["min_r"], pattern["max_r"]
        min_c, max_c = pattern["min_c"], pattern["max_c"]
        pattern_height = max_r - min_r + 1
        pattern_width = max_c - min_c + 1

        # Create pattern grid for tiling
        pattern_grid = [
            [0 for _ in range(pattern_width)] for _ in range(pattern_height)
        ]
        for r, c in pattern["cells"]:
            pattern_grid[r - min_r][c - min_c] = color

        # Place original pattern
        for r, c in pattern["cells"]:
            result[r][c] = color

        # Find vertical 1-lines that this pattern should connect to
        left_line = None
        right_line = None

        for line_c, line_start_r, line_end_r in vertical_1_lines:
            # Check if this vertical line overlaps with pattern rows
            if not (line_end_r < min_r or line_start_r > max_r):
                if line_c < min_c:
                    # This is a potential left line
                    if left_line is None or line_c > left_line:
                        left_line = line_c
                elif line_c > max_c:
                    # This is a potential right line
                    if right_line is None or line_c < right_line:
                        right_line = line_c

        # Fill horizontal extensions with tiled pattern
        if left_line is not None or right_line is not None:
            # Determine horizontal range
            start_c = left_line if left_line is not None else min_c
            end_c = right_line if right_line is not None else max_c

            for fill_r in range(min_r, max_r + 1):
                for fill_c in range(start_c, end_c + 1):
                    # Use tiled pattern
                    pattern_row = fill_r - min_r
                    pattern_col = (fill_c - min_c) % pattern_width
                    if pattern_grid[pattern_row][pattern_col] != 0:
                        result[fill_r][fill_c] = pattern_grid[pattern_row][pattern_col]

        # Find horizontal 1-lines that this pattern should connect to
        top_line = None
        bottom_line = None

        for line_r, line_start_c, line_end_c in horizontal_1_lines:
            # Check if this horizontal line overlaps with pattern columns
            if not (line_end_c < min_c or line_start_c > max_c):
                if line_r < min_r:
                    # This is a potential top line
                    if top_line is None or line_r > top_line:
                        top_line = line_r
                elif line_r > max_r:
                    # This is a potential bottom line
                    if bottom_line is None or line_r < bottom_line:
                        bottom_line = line_r

        # Fill vertical extensions with tiled pattern
        if top_line is not None or bottom_line is not None:
            # Determine vertical range
            start_r = top_line if top_line is not None else min_r
            end_r = bottom_line if bottom_line is not None else max_r

            for fill_r in range(start_r, end_r + 1):
                for fill_c in range(min_c, max_c + 1):
                    # Use tiled pattern
                    pattern_row = (fill_r - min_r) % pattern_height
                    pattern_col = fill_c - min_c
                    if pattern_grid[pattern_row][pattern_col] != 0:
                        result[fill_r][fill_c] = pattern_grid[pattern_row][pattern_col]

    return result
