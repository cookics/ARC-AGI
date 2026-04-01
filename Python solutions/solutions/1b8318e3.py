def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 2x2 blocks of 5s that remain fixed
    2. Non-zero, non-5 values move to adjacent positions (cardinal OR diagonal) around blocks
    3. Key: Process values in a specific order to avoid conflicts
    4. Strategy: For each value, find the globally closest adjacent position across all blocks

    Procedure:
    1. Identify all 2x2 blocks of 5s
    2. Sort values by distance to their nearest block (closest first)
    3. Assign each value greedily to its best available position
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all 2x2 blocks of 5s and copy to result
    blocks = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if (
                grid[r][c] == 5
                and grid[r][c + 1] == 5
                and grid[r + 1][c] == 5
                and grid[r + 1][c + 1] == 5
            ):
                blocks.append((r, c))
                result[r][c] = 5
                result[r][c + 1] = 5
                result[r + 1][c] = 5
                result[r + 1][c + 1] = 5

    # Helper: get all adjacent positions to a block
    def get_adjacent_positions(br, bc):
        positions = []
        for dr in range(-1, 3):
            for dc in range(-1, 3):
                ar, ac = br + dr, bc + dc
                if not (0 <= ar < rows and 0 <= ac < cols):
                    continue
                if 0 <= dr <= 1 and 0 <= dc <= 1:  # Inside block
                    continue
                # Check adjacency
                is_adjacent = False
                for bdr in range(2):
                    for bdc in range(2):
                        if abs(ar - (br + bdr)) <= 1 and abs(ac - (bc + bdc)) <= 1:
                            is_adjacent = True
                            break
                    if is_adjacent:
                        break
                if is_adjacent:
                    positions.append((ar, ac))
        return positions

    # Collect values with their best possible distance
    value_data = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5:
                # Find minimum possible distance to any adjacent position
                min_dist = float("inf")
                for br, bc in blocks:
                    for pr, pc in get_adjacent_positions(br, bc):
                        dist = ((r - pr) ** 2 + (c - pc) ** 2) ** 0.5
                        min_dist = min(min_dist, dist)
                value_data.append((min_dist, r, c, grid[r][c]))

    # Sort by minimum distance (process closest values first)
    value_data.sort()

    # Assign each value to its best available position
    for _, orig_r, orig_c, val in value_data:
        best_pos = None
        best_dist = float("inf")

        # Try all blocks and all their adjacent positions
        for br, bc in blocks:
            for pr, pc in get_adjacent_positions(br, bc):
                if result[pr][pc] != 0:  # Already occupied
                    continue
                dist = ((orig_r - pr) ** 2 + (orig_c - pc) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pos = (pr, pc)

        # Place the value
        if best_pos:
            result[best_pos[0]][best_pos[1]] = val

    return result
