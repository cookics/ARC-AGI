def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a 3×3 block of 3s with a center value 2
    2. Input has scattered 9s throughout the grid
    3. 6s form rectangular frames around regions
    4. 9s from specific regions (inside/near the frame) are collected
    5. Collected 9s fill the 3×3 pattern in reading order
    6. If 9 9s are collected, fill all cells including the center

    Procedure:
    1. Find the 3×3 pattern with center 2
    2. Find the frame region (bounding box of 6s)
    3. Collect 9s from frame rows and pattern columns above frame
    4. Fill the 3×3 pattern with collected 9s in reading order
    """
    import copy

    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find the 3×3 pattern (center value 2)
    pattern_r, pattern_c = None, None
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] == 2:
                # Check if surrounded by 3s
                if all(grid[r+dr][c+dc] == 3
                       for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                       if (dr, dc) != (0, 0)):
                    pattern_r, pattern_c = r, c
                    break
        if pattern_r is not None:
            break

    if pattern_r is None:
        return result

    # Pattern area cells
    pattern_cells = set()
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            pattern_cells.add((pattern_r + dr, pattern_c + dc))

    # Pattern columns
    pattern_cols = set(range(pattern_c - 1, pattern_c + 2))

    # Find bounding box of 6s
    sixes = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 6]

    # Collect 9s based on their location
    collected_nines = []

    if not sixes:
        # No frame: collect all 9s
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 9:
                    collected_nines.append((r, c))
    else:
        # Check if pattern area is completely enclosed by 6s
        # Do a flood-fill and see if it reaches the grid edges
        from collections import deque

        temp_reachable = set()
        temp_queue = deque()
        for r, c in pattern_cells:
            temp_queue.append((r, c))
            temp_reachable.add((r, c))

        reaches_edge = False
        while temp_queue:
            r, c = temp_queue.popleft()
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                reaches_edge = True
                # Continue to build full reachable set
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in temp_reachable and grid[nr][nc] != 6:
                        temp_reachable.add((nr, nc))
                        temp_queue.append((nr, nc))

        if not reaches_edge:
            # Pattern is completely enclosed - collect all 9s in same room
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 9 and (r, c) in temp_reachable and (r, c) not in pattern_cells:
                        collected_nines.append((r, c))
        else:
            # Pattern is not completely enclosed - use frame-based collection
            min_r = min(r for r, c in sixes)
            max_r = max(r for r, c in sixes)

            # Collect 9s from:
            # 1. Frame rows (min_r to max_r)
            # 2. Pattern columns close to frame (within 2 rows above)
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 9 and (r, c) not in pattern_cells:
                        in_frame_rows = min_r <= r <= max_r
                        in_pattern_cols_near_frame = (c in pattern_cols and r >= min_r - 2)

                        if in_frame_rows or in_pattern_cols_near_frame:
                            collected_nines.append((r, c))

    # Remove collected 9s from original positions
    for r, c in collected_nines:
        result[r][c] = 7

    # Fill the 3×3 pattern with 9s in reading order
    fill_positions = []
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            fill_positions.append((pattern_r + dr, pattern_c + dc))

    # Determine how many cells to fill
    num_to_fill = min(len(collected_nines), 9)

    # If we have exactly 9 9s, fill all 9 cells
    # Otherwise, skip the center (preserve the 2)
    if num_to_fill == 9:
        for i in range(9):
            r, c = fill_positions[i]
            result[r][c] = 9
    else:
        filled = 0
        for r, c in fill_positions:
            if filled >= num_to_fill:
                break
            if (r, c) != (pattern_r, pattern_c):  # Skip center
                result[r][c] = 9
                filled += 1

    return result
