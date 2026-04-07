def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a background color (most common), a pattern made of 0s, and a unique marker
    2. Output replicates the 0 pattern in the direction indicated by the marker
    3. Pattern is replicated as complete blocks shifted by the direction vector
    4. For small diagonal patterns: replicate multiple times along diagonal
    5. For large symmetric patterns: single reflection across marker

    Procedure:
    1. Find background color, 0 positions, and marker
    2. Calculate shift vector from a reference 0 to the marker
    3. Replicate pattern by shifting all 0s by multiples of shift vector
    4. Place marker color at replicated positions
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find background color (most common value)
    from collections import Counter
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_values).most_common(1)[0][0]

    # Find all 0 positions
    zero_positions = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
    if not zero_positions:
        return result

    # Find marker (unique non-background, non-0 value)
    marker_value, marker_pos = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and grid[r][c] != 0:
                marker_value, marker_pos = grid[r][c], (r, c)
                break
        if marker_value:
            break

    if not marker_value:
        return result

    # Pattern bounds
    min_r = min(r for r, c in zero_positions)
    max_r = max(r for r, c in zero_positions)
    min_c = min(c for r, c in zero_positions)
    max_c = max(c for r, c in zero_positions)
    marker_r, marker_c = marker_pos

    # Find actual 0 closest to marker
    best_zero = min(zero_positions, key=lambda p: abs(p[0] - marker_r) + abs(p[1] - marker_c))

    shift_r = marker_r - best_zero[0]
    shift_c = marker_c - best_zero[1]

    # Group 0s by row and identify center elements
    zeros_by_row = {}
    for r, c in zero_positions:
        if r not in zeros_by_row:
            zeros_by_row[r] = []
        zeros_by_row[r].append(c)

    center_zeros = set()
    for r, row_cols in zeros_by_row.items():
        if len(row_cols) >= 3:  # Wide rows have center elements
            sorted_cols = sorted(row_cols)
            center_zeros.add((r, sorted_cols[len(sorted_cols) // 2]))

    # Determine direction from pattern to marker
    dr = 1 if shift_r > 0 else -1 if shift_r < 0 else 0
    dc = 1 if shift_c > 0 else -1 if shift_c < 0 else 0

    # Determine row boundary (don't place markers before this boundary)
    if dr > 0:
        row_boundary = marker_r + 1  # For downward, start after marker
    elif dr < 0:
        row_boundary = marker_r - 1  # For upward, start before marker
    else:
        row_boundary = None

    # Check if this is a large symmetric pattern requiring block reflection
    pattern_height = max_r - min_r + 1
    is_large_pattern = pattern_height >= 5
    is_upward = dr < 0

    if is_large_pattern and is_upward:
        # Block reflection for large upward patterns (like example 3)
        # Reflect the entire pattern to the other side of the marker
        for r, c in zero_positions:
            # Calculate position in reflected pattern
            # Pattern rows 5-9 reflect to rows 4-0 (reversed)
            # Row 5 (min_r) -> row 4 (marker_r)
            # Row 9 (max_r) -> row 0 (marker_r - pattern_height + 1)
            new_r = marker_r - (r - min_r)
            # Pattern cols 6-10 shift right to align with marker at col 11
            new_c = c + (marker_c - min_c)

            if 0 <= new_r < rows and 0 <= new_c < cols:
                if grid[new_r][new_c] != 0:
                    result[new_r][new_c] = marker_value
    else:
        # Ray-based replication for smaller/diagonal patterns
        max_replications = max(rows, cols)
        for k in range(1, max_replications):
            for r, c in zero_positions:
                new_r = r + k * shift_r
                new_c = c + k * shift_c

                if not (0 <= new_r < rows and 0 <= new_c < cols):
                    continue

                if grid[new_r][new_c] == 0:  # Don't overwrite other 0s
                    continue

                # Skip if before/after the boundary
                if dr > 0 and new_r < row_boundary:
                    continue
                elif dr < 0 and new_r > row_boundary:
                    continue

                # Skip center elements in odd-numbered replications
                if (r, c) in center_zeros and k % 2 == 1:
                    continue

                result[new_r][new_c] = marker_value

    return result
