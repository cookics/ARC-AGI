def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Source pattern of 2s
    2. Horizontal 4-0-4 pairs
    3. Single 4s not in pairs
    4. For each single, find nearest pair sharing column
    5. Place pattern at midpoint between them

    Procedure:
    1. Extract pattern from 2s
    2. Find 4-0-4 pairs
    3. Find single 4s
    4. For single+pair combos, place pattern at geometric center
    """

    height, width = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Get source pattern
    twos = [(r, c) for r in range(height) for c in range(width) if grid[r][c] == 2]
    if not twos:
        return result

    min_r, max_r = min(r for r, c in twos), max(r for r, c in twos)
    min_c, max_c = min(c for r, c in twos), max(c for r, c in twos)
    pattern = [(r - min_r, c - min_c) for r, c in twos]
    ph, pw = max_r - min_r + 1, max_c - min_c + 1

    # Find 4-0-4 horizontal pairs
    pairs = [(r, c) for r in range(height) for c in range(width - 2)
             if grid[r][c] == 4 and grid[r][c+1] == 0 and grid[r][c+2] == 4]

    # Find single 4s
    all_fours = {(r, c) for r in range(height) for c in range(width) if grid[r][c] == 4}
    pair_fours = {(r, c+i) for r, c in pairs for i in [0, 2]}
    singles = all_fours - pair_fours

    # For each single, find pair in same column and place pattern
    for sr, sc in singles:
        # Find matching pair
        matches = [(abs(pr-sr), pr, pc) for pr, pc in pairs if sc in [pc, pc+1, pc+2]]
        if not matches:
            continue

        matches.sort()
        _, pr, pc = matches[0]

        # Determine column alignment
        col_off = sc - pc  # 0=left, 1=mid, 2=right

        # Calculate pattern placement
        # Key: pattern placed between single and pair
        if sr < pr:  # Single above pair
            # Pattern bottom aligns near single, shifted based on column
            if col_off == 0:  # Left align
                tr, tc = sr - ph + 1, pc + 2
                flip = True
            elif col_off == 2:  # Right align
                tr, tc = sr - ph + 1, pc + pw + 10
                flip = True
            else:  # Middle
                tr, tc = pr - ph - 1, pc + 2
                flip = True
        else:  # Single below pair
            # Pattern top aligns near pair
            if col_off == 0:  # Left
                tr, tc = pr, pc - pw - 2
                flip = True
            elif col_off == 2:  # Right
                tr, tc = pr, pc + 2
                flip = True
            else:  # Middle
                tr, tc = pr + 1, pc + 2
                flip = False

        # Place pattern
        for dr, dc in pattern:
            r_idx = ph - 1 - dr if flip else dr
            nr, nc = tr + r_idx, tc + dc
            if 0 <= nr < height and 0 <= nc < width:
                result[nr][nc] = 2

    return result
