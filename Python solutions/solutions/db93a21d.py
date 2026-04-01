def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has rectangular blocks of 9s
    2. Each block gets a frame of 3s (width 1 for 2x2, width 2 for larger)
    3. Vertical lines of 1s extend down from each block
    4. In meeting rows (where frames overlap vertically), horizontal bands connect structures
    5. Priority: 9 > 3 > 1 > 0

    Procedure:
    1. Find blocks, compute frames
    2. Identify meeting rows (where frames of consecutive blocks overlap/touch vertically)
    3. Place 9s first
    4. Draw vertical lines
    5. Draw frames
    6. In meeting rows, fill horizontal bands from leftmost structure to rightmost (except 9s and leftmost vertical line column)
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all blocks
    blocks = []
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 9 and not visited[r][c]:
                min_r, max_r, min_c, max_c = r, r, c, c
                queue = [(r, c)]
                visited[r][c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] == 9 and not visited[nr][nc]:
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                height = max_r - min_r + 1
                width = max_c - min_c + 1
                border_width = 1 if (height <= 2 and width <= 2) else 2

                blocks.append((min_r, max_r, min_c, max_c, border_width))

    if not blocks:
        return result

    blocks.sort()

    # Compute frames
    frames = []
    for r1, r2, c1, c2, bw in blocks:
        fr_r1 = max(0, r1 - bw)
        fr_r2 = min(rows - 1, r2 + bw)
        fr_c1 = max(0, c1 - bw)
        fr_c2 = min(cols - 1, c2 + bw)
        frames.append((fr_r1, fr_r2, fr_c1, fr_c2, r1, r2, c1, c2))

    # Identify meeting rows
    meeting_rows = set()
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            fr1_r1, fr1_r2, fr1_c1, fr1_c2, _, _, _, _ = frames[i]
            fr2_r1, fr2_r2, fr2_c1, fr2_c2, _, _, _, _ = frames[j]

            # Check if frames overlap or are adjacent vertically
            if not (fr1_r2 < fr2_r1 - 1 or fr2_r2 < fr1_r1 - 1):
                # They overlap or are within 1 row
                overlap_start = max(fr1_r1, fr2_r1 - 1)
                overlap_end = min(fr1_r2, fr2_r2)
                for r in range(overlap_start, overlap_end + 1):
                    meeting_rows.add(r)

    # Place 9s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 9:
                result[r][c] = 9

    # Draw vertical lines
    for r1, r2, c1, c2, bw in blocks:
        start_row = r2 + bw + 1
        for r in range(start_row, rows):
            for c in range(c1, c2 + 1):
                if result[r][c] == 0:
                    result[r][c] = 1

    # Draw frames
    for fr_r1, fr_r2, fr_c1, fr_c2, b_r1, b_r2, b_c1, b_c2 in frames:
        # Top border
        for r in range(fr_r1, b_r1):
            for c in range(fr_c1, fr_c2 + 1):
                if result[r][c] != 9:
                    result[r][c] = 3

        # Bottom border
        for r in range(b_r2 + 1, fr_r2 + 1):
            for c in range(fr_c1, fr_c2 + 1):
                if result[r][c] != 9:
                    result[r][c] = 3

        # Side borders
        for r in range(fr_r1, fr_r2 + 1):
            for c in range(fr_c1, b_c1):
                if result[r][c] != 9:
                    result[r][c] = 3
            for c in range(b_c2 + 1, fr_c2 + 1):
                if result[r][c] != 9:
                    result[r][c] = 3

    # Fill horizontal bands in meeting rows
    for r in meeting_rows:
        non_zero = [c for c in range(cols) if result[r][c] != 0]
        if len(non_zero) >= 2:
            # Find leftmost vertical line (value 1)
            leftmost_vline = None
            for c in non_zero:
                if result[r][c] == 1:
                    leftmost_vline = c
                    break

            # Fill from (leftmost_vline + 1) or first non-zero to rightmost non-zero
            left = (leftmost_vline + 1) if leftmost_vline is not None else non_zero[0]
            right = non_zero[-1]

            for c in range(left, right + 1):
                if result[r][c] != 9:
                    result[r][c] = 3

    return result
