def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Markers form diagonal lines (r+c=constant, slope -1)
    2. For each row, process markers to determine fill patterns
    3. Each marker triggers fills based on its diagonal overlap with neighbors
    4. Fill widths grow with row offset: 2 * (row - overlap_start)

    Procedure:
    1. Group markers by diagonal and track extents
    2. For each row, find markers and their diagonal contexts
    3. For each marker, fill cells to the right based on overlap calculations
    4. Handle multiple diagonal pairs per row
    """
    n = len(grid)
    m = len(grid[0])

    # Find markers and group by diagonal
    diagonals = {}
    for r in range(n):
        for c in range(m):
            if grid[r][c] != 8:
                diag_id = r + c
                if diag_id not in diagonals:
                    diagonals[diag_id] = []
                diagonals[diag_id].append((r, c))

    # Track vertical extent
    diag_info = {}
    for diag_id, cells in diagonals.items():
        cells.sort()
        diag_info[diag_id] = (cells[0][0], cells[-1][0])

    result = [row[:] for row in grid]
    sorted_diags = sorted(diag_info.keys())

    # For each row, process markers
    for r in range(n):
        # Find which diagonals cross this row
        row_diags = []
        for diag_id in sorted_diags:
            col = diag_id - r
            if 0 <= col < m:
                row_diags.append((col, diag_id))

        row_diags.sort()

        # Process each marker
        for idx, (col, diag_id) in enumerate(row_diags):
            if grid[r][col] == 8:  # Skip if not an actual marker
                continue

            # Find next diagonal
            if idx + 1 < len(row_diags):
                next_col, next_diag_id = row_diags[idx + 1]

                r_min1, r_max1 = diag_info[diag_id]
                r_min2, r_max2 = diag_info[next_diag_id]

                # Calculate overlap
                overlap_start = max(r_min1, r_min2)
                overlap_end = min(r_max1, r_max2)

                if overlap_start <= overlap_end and r >= overlap_start:
                    row_offset = r - overlap_start
                    # Skip first row only if it's the absolute first row for both diagonals
                    first_row_overall = min(r_min1, r_min2)
                    if row_offset > 0 or overlap_start > first_row_overall:
                        fill_width = 2 * row_offset if row_offset > 0 else 2
                        for c in range(col + 1, min(col + 1 + fill_width, next_col, m)):
                            if result[r][c] == 8:
                                result[r][c] = 2

    return result
