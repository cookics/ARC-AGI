def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Input has a vertical line and horizontal line that intersect.
    Output creates a rectangular pattern with alternating colors.

    Procedure:
    1. Find the main vertical and horizontal lines
    2. Identify main and accent colors
    3. Create pattern based on simple alternating rules
    """
    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find horizontal and vertical lines
    h_row = max(
        range(rows), key=lambda r: sum(1 for c in range(cols) if grid[r][c] != 0)
    )
    v_col = max(
        range(cols), key=lambda c: sum(1 for r in range(rows) if grid[r][c] != 0)
    )

    # Get main colors
    from collections import Counter

    h_counter = Counter(grid[h_row][c] for c in range(cols) if grid[h_row][c] != 0)
    v_counter = Counter(grid[r][v_col] for r in range(rows) if grid[r][v_col] != 0)

    h_main = h_counter.most_common(1)[0][0]
    v_main = v_counter.most_common(1)[0][0]

    # Find accent positions and colors
    h_accents = {}
    for c in range(cols):
        if grid[h_row][c] != 0 and grid[h_row][c] != h_main:
            h_accents[c] = grid[h_row][c]

    v_accents = {}
    for r in range(rows):
        if grid[r][v_col] != 0 and grid[r][v_col] != v_main:
            v_accents[r] = grid[r][v_col]

    if not h_accents:
        return result

    # Pattern boundaries
    start_col = min(h_accents.keys())
    end_col = max(h_accents.keys())
    pattern_width = end_col - start_col + 1

    if v_accents:
        start_row = min(v_accents.keys())
    else:
        start_row = 1
    end_row = h_row

    # Get accent colors
    v_accent_color = list(v_accents.values())[0] if v_accents else None
    h_accent_color = list(h_accents.values())[0] if h_accents else None

    # Create pattern
    for r in range(start_row, end_row + 1):
        for c in range(start_col, end_col + 1):
            col_offset = c - start_col

            if pattern_width == 3:
                # 3-column pattern (working correctly)
                if c == start_col or c == end_col:  # Side columns
                    if r % 2 == 1:  # Odd rows
                        if r >= end_row - 2:  # Close to horizontal line
                            result[r][c] = (
                                h_accent_color if h_accent_color else v_accent_color
                            )
                        else:
                            result[r][c] = v_accent_color if v_accent_color else h_main
                    else:  # Even rows
                        result[r][c] = h_main
                else:  # Center column
                    if r % 2 == 1:  # Odd rows
                        result[r][c] = v_main
                    else:  # Even rows
                        result[r][c] = h_main
            else:
                # Multi-column pattern - much simpler approach
                if r == h_row:
                    # Horizontal line: alternate h_accent and v_main
                    if col_offset % 2 == 0:
                        result[r][c] = h_accent_color
                    else:
                        result[r][c] = v_main
                else:
                    # Non-horizontal line rows
                    row_type = (
                        r - start_row
                    ) % 4  # 4-row cycle for multi-column patterns

                    if row_type == 1 or row_type == 3:
                        # Rows that should be all h_main
                        result[r][c] = h_main
                    else:
                        # Rows that should have alternating pattern
                        if col_offset % 2 == 0:
                            # Even columns: use accent colors
                            if r in v_accents:
                                result[r][c] = v_accent_color
                            else:
                                result[r][c] = (
                                    h_accent_color if h_accent_color else v_accent_color
                                )
                        else:
                            # Odd columns: varies by row
                            if r in v_accents:
                                # For rows with v_accent, use v_main in later parts
                                if r >= (start_row + end_row) * 2 // 3:
                                    result[r][c] = v_main
                                else:
                                    result[r][c] = h_main
                            else:
                                result[r][c] = v_main

    return result
