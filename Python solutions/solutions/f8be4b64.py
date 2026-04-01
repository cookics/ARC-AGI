def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has cross/diamond patterns (4 3s surrounding a center value)
    2. Crosses with non-zero, non-3 centers generate vertical and horizontal lines
    3. Crosses with center 0 block their columns for horizontal lines
    4. Horizontal lines have limited extent when crosses are close together (adjacent rows)

    Procedure:
    1. Find all cross patterns and preserve their positions
    2. Identify crosses with center 0 (blockers) and non-zero centers (generators)
    3. Group crosses that are adjacent (within 2 rows) to determine horizontal extent
    4. Draw horizontal lines with appropriate extent
    5. Draw vertical lines (full column)
    6. Preserve cross structures and centers
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all crosses
    all_cross_positions = set()
    blocked_columns = set()
    crosses_to_draw = []

    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if (
                grid[r - 1][c] == 3
                and grid[r + 1][c] == 3
                and grid[r][c - 1] == 3
                and grid[r][c + 1] == 3
            ):
                all_cross_positions.update(
                    [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1), (r, c)]
                )

                center = grid[r][c]
                if center == 0:
                    blocked_columns.add(c)
                elif center != 3:
                    crosses_to_draw.append((r, c, center))

    # Group adjacent crosses (within 2 rows of each other)
    crosses_to_draw.sort()  # Sort by row, then column

    def get_horizontal_extent(cross_idx):
        r, c, color = crosses_to_draw[cross_idx]

        # Find crosses within 2 rows
        adjacent = []
        for i, (r2, c2, _) in enumerate(crosses_to_draw):
            if abs(r2 - r) <= 2:
                adjacent.append((i, c2))

        if len(adjacent) <= 1:
            # No adjacent crosses, fill entire row
            return 0, cols - 1

        # Sort by column
        adjacent.sort(key=lambda x: x[1])
        cols_only = [c2 for _, c2 in adjacent]

        # Find position of current cross in adjacent list
        pos = cols_only.index(c)

        if pos == 0:
            # Leftmost in group: fill from 0 to next cross column
            return 0, cols_only[1]
        elif pos == len(cols_only) - 1:
            # Rightmost in group: fill from prev cross column + 1 to end
            return cols_only[-2] + 1, cols - 1
        else:
            # Middle: fill from prev + 1 to next
            return cols_only[pos - 1] + 1, cols_only[pos + 1]

    # Draw horizontal lines with extent
    for idx, (r, c, color) in enumerate(crosses_to_draw):
        start_col, end_col = get_horizontal_extent(idx)
        for col in range(start_col, end_col + 1):
            if (
                col not in blocked_columns
                and grid[r][col] == 0
                and (r, col) not in all_cross_positions
            ):
                result[r][col] = color

    # Draw vertical lines with limits based on overlapping diamonds
    for r, c, color in crosses_to_draw:
        # Find diamonds that overlap with this column (their cross touches column c)
        overlapping = []
        for r2, c2, _ in crosses_to_draw:
            # A cross at (r2, c2) spans columns c2-1, c2, c2+1
            if abs(c2 - c) <= 1:
                overlapping.append(r2)

        overlapping.sort()

        # Find range for vertical line
        idx = overlapping.index(r)
        start_row = overlapping[idx - 1] + 1 if idx > 0 else 0
        end_row = overlapping[idx + 1] - 1 if idx < len(overlapping) - 1 else rows - 1

        # Draw vertical line in the allowed range
        for row in range(start_row, end_row + 1):
            if grid[row][c] == 0 and (row, c) not in all_cross_positions:
                result[row][c] = color

    # Restore cross centers
    for r, c, color in crosses_to_draw:
        result[r][c] = color

    return result
