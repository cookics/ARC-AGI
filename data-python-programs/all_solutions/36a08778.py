def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background 7s, vertical marker 6s, and horizontal target 2s
    2. Vertical 6 markers extend downward until hitting a 2
    3. Horizontal sequences of 3+ consecutive 2s get rectangular border frames of 6s
    4. Segments on rows with seed columns don't get frames
    5. Segments under a seed-row segment (within its column span) don't get frames

    Procedure:
    1. Find all horizontal sequences of consecutive 2s (length >= 3)
    2. Identify which segments should get frames (not on seed rows, not under seed-row segments)
    3. Draw border frames around valid segments
    4. Extend vertical edges of frames downward to connect to next frame
    5. Extend original vertical 6 columns downward until hitting a 2
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    result = [row[:] for row in grid]

    # Step 1: Find all horizontal sequences of 2s (length >= 3)
    sequences = []
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] == 2:
                start = c
                while c < cols and grid[r][c] == 2:
                    c += 1
                end = c - 1
                if end - start + 1 >= 3:
                    sequences.append((r, start, end))
            else:
                c += 1

    # Find seed columns (columns with at least one 6)
    seed_cols = set()
    for c in range(cols):
        if any(grid[r][c] == 6 for r in range(rows)):
            seed_cols.add(c)

    # Step 2: Find seed-row segments (segments on rows where seed columns pass through)
    seed_row_segments = []
    for r, start, end in sequences:
        # Check if any seed column passes through this row (anywhere on the row)
        has_seed_passing = False
        for sc in seed_cols:
            seed_start_row = None
            for sr in range(rows):
                if grid[sr][sc] == 6:
                    seed_start_row = sr
                    break
            if seed_start_row is not None:
                # Check if seed extends to current row (stops before hitting a 2)
                extends_to_row = True
                for check_r in range(seed_start_row, r):
                    if grid[check_r][sc] == 2:
                        extends_to_row = False
                        break
                if extends_to_row:
                    has_seed_passing = True
                    break
        if has_seed_passing:
            seed_row_segments.append((r, start, end))

    # Step 3: Determine which segments should get frames
    valid_sequences = []
    for r, start, end in sequences:
        # Skip if seed column passes through this row (anywhere on the row)
        has_seed_passing = False
        for sc in seed_cols:
            seed_start_row = None
            for sr in range(rows):
                if grid[sr][sc] == 6:
                    seed_start_row = sr
                    break
            if seed_start_row is not None:
                # Check if seed extends to current row (stops before hitting a 2)
                extends_to_row = True
                for check_r in range(seed_start_row, r):
                    if grid[check_r][sc] == 2:
                        extends_to_row = False
                        break
                if extends_to_row:
                    has_seed_passing = True
                    break
        if has_seed_passing:
            continue

        # Skip if completely within a seed-row segment's column span (and close enough vertically)
        within_seed_segment = False
        for sr, sstart, send in seed_row_segments:
            if sr < r and r - sr <= 3:  # seed row is above and within distance 3
                # Check if segment is COMPLETELY within seed-row segment span
                if start >= sstart and end <= send:
                    within_seed_segment = True
                    break
        if within_seed_segment:
            continue

        valid_sequences.append((r, start, end))

    # Step 4: Draw border frames and extend edges
    for r, start, end in valid_sequences:
        # Frame boundaries
        top = r - 1
        bottom = r + 1
        left = start - 1
        right = end + 1

        # Draw top edge
        if top >= 0:
            for c in range(max(0, left), min(cols, right + 1)):
                if result[top][c] != 2:
                    result[top][c] = 6

        # Draw and extend left edge
        if left >= 0:
            # Draw left edge for rows r and r+1
            if result[r][left] != 2:
                result[r][left] = 6
            if bottom < rows and result[bottom][left] != 2:
                result[bottom][left] = 6

            # Extend left edge downward
            for row in range(bottom + 1, rows):
                if result[row][left] == 2:
                    break
                # Stop if we hit a frame top edge
                hit_top = False
                for r2, start2, end2 in valid_sequences:
                    if r2 - 1 == row and start2 - 1 <= left <= end2 + 1:
                        hit_top = True
                        break
                if hit_top:
                    break
                result[row][left] = 6

        # Draw and extend right edge
        if right < cols:
            # Draw right edge for rows r and r+1
            if result[r][right] != 2:
                result[r][right] = 6
            if bottom < rows and result[bottom][right] != 2:
                result[bottom][right] = 6

            # Extend right edge downward
            for row in range(bottom + 1, rows):
                if result[row][right] == 2:
                    break
                # Stop if we hit a frame top edge
                hit_top = False
                for r2, start2, end2 in valid_sequences:
                    if r2 - 1 == row and start2 - 1 <= right <= end2 + 1:
                        hit_top = True
                        break
                if hit_top:
                    break
                result[row][right] = 6

    # Step 5: Extend original vertical 6 columns downward
    for c in range(cols):
        for r in range(rows):
            if grid[r][c] == 6:
                # Extend downward from this original 6
                for r2 in range(r + 1, rows):
                    if result[r2][c] == 2:
                        break
                    result[r2][c] = 6

    return result
