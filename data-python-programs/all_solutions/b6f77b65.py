def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid contains hollow rectangular frames stacked vertically
    2. Marker encodes shift; marker color is removed
    3. Frames near bottom stay grounded, others shift
    4. Grounded determined by frame connectivity to bottom rows

    Procedure:
    1. Parse marker, compute shift
    2. Remove marker color
    3. Detect hollow rectangular frames
    4. Classify frames as grounded vs shifting
    5. Apply transformation with overlaps resolved
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    marker1 = grid[0][0]
    marker2 = grid[0][1] if cols > 1 else 0

    shift_map = {9: 0, 7: 5, 6: 2, 4: 3}

    if marker2 != 0:
        shift = marker1 + marker2
    else:
        shift = shift_map.get(marker1, 0)

    if shift == 0:
        return [row[:] for row in grid]

    marker = marker1

    # Remove marker color
    cleaned = [row[:] for row in grid]
    for i in range(2, rows):
        for j in range(cols):
            if grid[i][j] == marker:
                cleaned[i][j] = 0

    # Find hollow rectangular frames
    def find_frames():
        frames = []
        # Try to find frames by detecting rectangular borders
        for r1 in range(2, rows-2):
            for c1 in range(cols-2):
                for r2 in range(r1+2, rows):
                    for c2 in range(c1+2, cols):
                        # Check if (r1,c1) to (r2,c2) forms a hollow frame
                        # Top/bottom borders should be non-zero, interior mostly zero
                        top_vals = [cleaned[r1][c] for c in range(c1, c2+1) if cleaned[r1][c] != 0]
                        bot_vals = [cleaned[r2][c] for c in range(c1, c2+1) if cleaned[r2][c] != 0]
                        left_vals = [cleaned[r][c1] for r in range(r1, r2+1) if cleaned[r][c1] != 0]
                        right_vals = [cleaned[r][c2] for r in range(r1, r2+1) if cleaned[r][c2] != 0]

                        if (len(top_vals) >= 2 and len(bot_vals) >= 2 and
                            len(left_vals) >= 2 and len(right_vals) >= 2):
                            # Check interior is mostly empty
                            interior_count = 0
                            for r in range(r1+1, r2):
                                for c in range(c1+1, c2):
                                    if cleaned[r][c] != 0:
                                        interior_count += 1
                            if interior_count < (r2-r1-1) * (c2-c1-1) * 0.3:  # <30% filled
                                frames.append((r1, r2, c1, c2))

        # Remove overlapping frames (keep largest)
        frames.sort(key=lambda f: (f[1]-f[0])*(f[3]-f[2]), reverse=True)
        filtered = []
        for f in frames:
            overlap = False
            for existing in filtered:
                if not (f[1] < existing[0] or f[0] > existing[1] or
                       f[3] < existing[2] or f[2] > existing[3]):
                    overlap = True
                    break
            if not overlap:
                filtered.append(f)

        return filtered

    frames = find_frames()

    # Classify frames: those touching/near row 11 are grounded
    grounded_frames = []
    shifting_frames = []
    for frame in frames:
        r1, r2, c1, c2 = frame
        if r2 >= rows - 3:  # Bottom within 3 rows of end
            grounded_frames.append(frame)
        else:
            shifting_frames.append(frame)

    # Build result
    result = [[0] * cols for _ in range(rows)]
    result[0] = grid[0][:]
    result[1] = grid[1][:]

    # Place grounded frames first
    for r1, r2, c1, c2 in grounded_frames:
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                if cleaned[r][c] != 0:
                    result[r][c] = cleaned[r][c]

    # Shift and place shifting frames
    for r1, r2, c1, c2 in shifting_frames:
        for r in range(r1, r2+1):
            new_r = r + shift
            if new_r < rows:
                for c in range(c1, c2+1):
                    if cleaned[r][c] != 0 and result[new_r][c] == 0:
                        result[new_r][c] = cleaned[r][c]

    # Place any remaining non-frame cells
    for i in range(2, rows):
        for j in range(cols):
            if cleaned[i][j] != 0 and result[i][j] == 0:
                new_i = i + shift
                if new_i < rows:
                    result[new_i][j] = cleaned[i][j]

    return result
