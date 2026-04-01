def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has diagonal boundary between two background colors
    2. 2x2 blocks define pattern (keeping ALL blocks in order, not just unique values!)
    3. Last block is terminator
    4. Vertical line uses rotated pattern, horizontal uses original
    5. Draw patterns at boundary column and marker rows

    Procedure:
    1. Extract full pattern from blocks (keep duplicates)
    2. Erase blocks
    3. Find main boundary column
    4. Draw vertical with rotated pattern
    5. Draw horizontal at markers and boundary crossings
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Find two most common colors
    flat = [cell for row in grid for cell in row]
    counts = Counter(flat)
    bg = sorted(counts.keys(), key=lambda c: counts[c], reverse=True)[:2]
    bg_set = set(bg)

    # Find ALL 2x2 blocks
    blocks = []
    block_cells = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            val = grid[r][c]
            if val not in bg_set and (r, c) not in block_cells:
                if grid[r][c+1] == val and grid[r+1][c] == val and grid[r+1][c+1] == val:
                    blocks.append((r, c, val))
                    block_cells.update([(r,c), (r,c+1), (r+1,c), (r+1,c+1)])

    if not blocks:
        return grid

    # Sort and extract pattern
    blocks_sorted = sorted(blocks, key=lambda b: (b[0], b[1]))
    term = blocks_sorted[-1][2]
    pattern = [b[2] for b in blocks_sorted[:-1]]  # Keep ALL values including duplicates!

    if not pattern:
        return grid

    # Rotate pattern for vertical (move last to front)
    vert_pat = [pattern[-1]] + pattern[:-1]

    # Find markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in bg_set and (r, c) not in block_cells:
                markers.append((r, c))

    # Erase blocks
    result = [row[:] for row in grid]
    for br, bc, _ in blocks:
        for dr in range(2):
            for dc in range(2):
                r, c = br + dr, bc + dc
                nbrs = [result[nr][nc] for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                        if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] in bg_set]
                result[r][c] = Counter(nbrs).most_common(1)[0][0] if nbrs else bg[0]

    # Find ALL boundary information BEFORE drawing anything
    boundary_cols = []
    for r in range(rows):
        for c in range(cols - 1):
            if result[r][c] in bg_set and result[r][c+1] in bg_set:
                if result[r][c] != result[r][c+1]:
                    boundary_cols.append(c)
                    break

    if not boundary_cols:
        return result

    # Find marker row boundaries BEFORE modifying grid
    marker_boundaries = {}
    for mr, mc in markers:
        if mc == 0:
            for c in range(cols - 1):
                if result[mr][c] in bg_set and result[mr][c+1] in bg_set:
                    if result[mr][c] != result[mr][c+1]:
                        marker_boundaries[mr] = c
                        break

    # Find significant boundary columns
    col_counts = Counter(boundary_cols)
    significant_cols = [c for c, count in col_counts.items() if count >= rows // 4]
    significant_cols.sort()

    if not significant_cols:
        return result

    # Draw vertical patterns on each significant boundary column
    for col in significant_cols:
        # Find where vertical starts for this column
        v_start = 0
        for r in range(rows):
            if r < rows - 1:
                if result[r][col] in bg_set and result[r+1][col] in bg_set:
                    if result[r][col] != result[r+1][col]:
                        v_start = r
                        break

        # Draw vertical pattern on this column
        result[v_start][col] = term
        for i in range(1, rows - v_start):
            r = v_start + i
            if r < rows:
                result[r][col] = vert_pat[(i - 1) % len(vert_pat)]

    # Draw horizontal at marker rows using pre-computed boundaries
    for mr in marker_boundaries:
        row_boundary = marker_boundaries[mr]
        # Draw from 0 to boundary
        for c in range(row_boundary + 1):
            result[mr][c] = pattern[c % len(pattern)]
        # Place terminator
        if row_boundary + 1 < cols:
            result[mr][row_boundary + 1] = term

    # Draw horizontal segments at rows with boundary crossings
    for r in range(rows):
        if r not in marker_boundaries:
            # Check if this row has a boundary transition
            # Use the first significant column
            if not significant_cols:
                continue
            c = significant_cols[0]
            if c > 0 and c + 1 < cols:
                if result[r][c-1] in bg_set and result[r][c+1] in bg_set:
                    if result[r][c-1] != result[r][c+1]:
                        # Draw from main_col to next boundary transition
                        end_c = c + 1
                        while end_c < cols and result[r][end_c] in bg_set:
                            end_c += 1

                        # Place pattern
                        for i, col in enumerate(range(c, min(end_c, cols))):
                            if col < end_c - 1:
                                result[r][col] = vert_pat[i % len(vert_pat)]
                            else:
                                result[r][col] = term
                                break

    return result
