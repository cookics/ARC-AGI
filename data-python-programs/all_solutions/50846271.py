def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 0s, 5s, and 2s
    2. 2s form lines (horizontal/vertical)
    3. 5s become 8s when between/adjacent to 2s
    4. Horizontal lines with gaps: fill gaps only, no extension
    5. Horizontal lines without gaps: extend both sides
    6. Vertical lines: fill gaps + extend upward (multiple cells until non-5)
    7. Don't extend if row already has 2s (cross pattern)

    Procedure:
    1. Process rows: find 2 lines, fill/extend based on gaps
    2. Process columns: fill gaps + extend upward until non-5
    3. Check for cross patterns to avoid conflicts
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    result = [row[:] for row in grid]

    # Check which rows have 2s (for cross pattern detection)
    rows_with_twos = set()
    for r in range(rows):
        if any(grid[r][c] == 2 for c in range(cols)):
            rows_with_twos.add(r)

    # Process rows
    for r in range(rows):
        twos = [c for c in range(cols) if grid[r][c] == 2]
        if len(twos) < 2:
            continue

        # Check if line has gaps
        min_c, max_c = min(twos), max(twos)
        has_gaps = (max_c - min_c + 1) > len(twos)

        if has_gaps:
            # Fill all gaps between first and last 2
            for c in range(min_c + 1, max_c):
                if grid[r][c] == 5:
                    result[r][c] = 8
        else:
            # No gaps: extend both sides
            if min_c > 0 and grid[r][min_c - 1] == 5:
                result[r][min_c - 1] = 8
            if max_c < cols - 1 and grid[r][max_c + 1] == 5:
                result[r][max_c + 1] = 8

    # Process columns
    for c in range(cols):
        twos = [r for r in range(rows) if grid[r][c] == 2]
        if len(twos) < 2:
            continue

        min_r, max_r = min(twos), max(twos)

        # Fill all gaps between first and last 2
        for r in range(min_r + 1, max_r):
            if grid[r][c] == 5:
                result[r][c] = 8

        # Extend upward (max 2 rows) only if first 2 is far enough from top edge
        if min_r >= 5:  # At least 5 rows from top
            r = min_r - 1
            ext_count = 0
            while r >= 0 and grid[r][c] == 5 and r not in rows_with_twos and ext_count < 2:
                result[r][c] = 8
                r -= 1
                ext_count += 1

        # Extend downward by 1 (only if not in a row with 2s)
        if max_r < rows - 1 and grid[max_r + 1][c] == 5 and (max_r + 1) not in rows_with_twos:
            result[max_r + 1][c] = 8

    return result
