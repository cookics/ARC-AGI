def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The pattern connects aligned rectangular blocks using bridges made of 8s.
    When blocks are aligned horizontally or vertically but separated by gaps,
    the output fills those gaps with 8s to create connections. The bridges use the middle
    portion of the overlapping range (excluding edges). Blocks of ANY color can be connected.

    Procedure:
    1. Find all rectangular blocks (regardless of color)
    2. For all pairs of blocks, check if they can be connected (aligned blocks)
    3. Create bridges of 8s between aligned blocks in the middle of their overlapping range
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the grid

    # Find all rectangular blocks for each color using flood fill
    def find_blocks():
        blocks_by_color = {}
        visited = [[False] * cols for _ in range(rows)]

        for r in range(rows):
            for c in range(cols):
                if not visited[r][c] and grid[r][c] != 0:
                    color = grid[r][c]

                    # Use flood fill to find all connected cells
                    cells = []
                    queue = [(r, c)]
                    visited[r][c] = True

                    while queue:
                        cr, cc = queue.pop(0)
                        cells.append((cr, cc))

                        # Check 4-connected neighbors
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < rows
                                and 0 <= nc < cols
                                and not visited[nr][nc]
                                and grid[nr][nc] == color
                            ):
                                visited[nr][nc] = True
                                queue.append((nr, nc))

                    # Get bounding box
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)

                    # Store the block
                    if color not in blocks_by_color:
                        blocks_by_color[color] = []
                    blocks_by_color[color].append((min_r, max_r, min_c, max_c))

        return blocks_by_color

    # Create bridges between aligned blocks
    def create_bridges(blocks_by_color):
        for color, blocks in blocks_by_color.items():
            # Check all pairs of blocks for this color
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    r1_min, r1_max, c1_min, c1_max = blocks[i]
                    r2_min, r2_max, c2_min, c2_max = blocks[j]

                    # Check if horizontally aligned (same row range)
                    if r1_min <= r2_max and r2_min <= r1_max:  # Overlapping rows
                        # Determine the overlapping row range
                        overlap_r_min = max(r1_min, r2_min)
                        overlap_r_max = min(r1_max, r2_max)

                        # Use middle rows (exclude first and last if possible)
                        if overlap_r_max - overlap_r_min >= 2:
                            bridge_r_min = overlap_r_min + 1
                            bridge_r_max = overlap_r_max - 1
                        else:
                            bridge_r_min = overlap_r_min
                            bridge_r_max = overlap_r_max

                        # Check if there's a gap between blocks horizontally
                        if c1_max < c2_min:  # Block 1 is left of block 2
                            gap_size = c2_min - c1_max - 1
                            # Only create bridge if gap is reasonable (not too large)
                            if gap_size <= 12:  # Max bridge length constraint
                                # Check if there are any other blocks of same color in between
                                has_intervening_block = False
                                for k in range(len(blocks)):
                                    if k != i and k != j:
                                        kr_min, kr_max, kc_min, kc_max = blocks[k]
                                        # Check if this block is horizontally between our two blocks
                                        # and overlaps with the bridge area
                                        if (
                                            kc_min > c1_max
                                            and kc_max < c2_min
                                            and kr_min <= bridge_r_max
                                            and kr_max >= bridge_r_min
                                        ):
                                            has_intervening_block = True
                                            break

                                if not has_intervening_block:
                                    # Fill the gap with 8s
                                    for r in range(bridge_r_min, bridge_r_max + 1):
                                        for c in range(c1_max + 1, c2_min):
                                            if (
                                                result[r][c] == 0
                                            ):  # Only fill empty cells
                                                result[r][c] = 8
                        elif c2_max < c1_min:  # Block 2 is left of block 1
                            gap_size = c1_min - c2_max - 1
                            # Only create bridge if gap is reasonable (not too large)
                            if gap_size <= 12:  # Max bridge length constraint
                                # Check if there are any other blocks of same color in between
                                has_intervening_block = False
                                for k in range(len(blocks)):
                                    if k != i and k != j:
                                        kr_min, kr_max, kc_min, kc_max = blocks[k]
                                        # Check if this block is horizontally between our two blocks
                                        # and overlaps with the bridge area
                                        if (
                                            kc_min > c2_max
                                            and kc_max < c1_min
                                            and kr_min <= bridge_r_max
                                            and kr_max >= bridge_r_min
                                        ):
                                            has_intervening_block = True
                                            break

                                if not has_intervening_block:
                                    # Fill the gap with 8s
                                    for r in range(bridge_r_min, bridge_r_max + 1):
                                        for c in range(c2_max + 1, c1_min):
                                            if (
                                                result[r][c] == 0
                                            ):  # Only fill empty cells
                                                result[r][c] = 8

                    # Check if vertically aligned (same column range)
                    if c1_min <= c2_max and c2_min <= c1_max:  # Overlapping columns
                        # Determine the overlapping column range
                        overlap_c_min = max(c1_min, c2_min)
                        overlap_c_max = min(c1_max, c2_max)

                        # Use middle columns (exclude first and last if possible)
                        if overlap_c_max - overlap_c_min >= 2:
                            bridge_c_min = overlap_c_min + 1
                            bridge_c_max = overlap_c_max - 1
                        else:
                            bridge_c_min = overlap_c_min
                            bridge_c_max = overlap_c_max

                        # Check if there's a gap between blocks vertically
                        if r1_max < r2_min:  # Block 1 is above block 2
                            gap_size = r2_min - r1_max - 1
                            # Only create bridge if gap is reasonable (not too large)
                            if gap_size <= 12:  # Max bridge length constraint
                                # Check if there are any other blocks of same color in between
                                has_intervening_block = False
                                for k in range(len(blocks)):
                                    if k != i and k != j:
                                        kr_min, kr_max, kc_min, kc_max = blocks[k]
                                        # Check if this block is vertically between our two blocks
                                        # and overlaps with the bridge area
                                        if (
                                            kr_min > r1_max
                                            and kr_max < r2_min
                                            and kc_min <= bridge_c_max
                                            and kc_max >= bridge_c_min
                                        ):
                                            has_intervening_block = True
                                            break

                                if not has_intervening_block:
                                    # Fill the gap with 8s
                                    for r in range(r1_max + 1, r2_min):
                                        for c in range(bridge_c_min, bridge_c_max + 1):
                                            if (
                                                result[r][c] == 0
                                            ):  # Only fill empty cells
                                                result[r][c] = 8
                        elif r2_max < r1_min:  # Block 2 is above block 1
                            gap_size = r1_min - r2_max - 1
                            # Only create bridge if gap is reasonable (not too large)
                            if gap_size <= 12:  # Max bridge length constraint
                                # Check if there are any other blocks of same color in between
                                has_intervening_block = False
                                for k in range(len(blocks)):
                                    if k != i and k != j:
                                        kr_min, kr_max, kc_min, kc_max = blocks[k]
                                        # Check if this block is vertically between our two blocks
                                        # and overlaps with the bridge area
                                        if (
                                            kr_min > r2_max
                                            and kr_max < r1_min
                                            and kc_min <= bridge_c_max
                                            and kc_max >= bridge_c_min
                                        ):
                                            has_intervening_block = True
                                            break

                                if not has_intervening_block:
                                    # Fill the gap with 8s
                                    for r in range(r2_max + 1, r1_min):
                                        for c in range(bridge_c_min, bridge_c_max + 1):
                                            if (
                                                result[r][c] == 0
                                            ):  # Only fill empty cells
                                                result[r][c] = 8

    # Execute the algorithm
    blocks_by_color = find_blocks()
    create_bridges(blocks_by_color)

    return result
