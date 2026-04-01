def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains connected components (4-connected) of value 3
    2. Output preserves the 3s and adds 8s in specific positions
    3. For diagonal pairs of 3s: place 8s on perpendicular diagonal forming a square (all 4 points equidistant)
    4. For rectangular blocks: when there are exactly 2 blocks, place 8-blocks perpendicular to the line connecting centers

    Procedure:
    1. Find all connected components of 3s (using 4-connectivity)
    2. Check for diagonal pairs (two 3s that are diagonally adjacent but not 4-connected)
    3. For each diagonal pair: place 8s on the perpendicular diagonal
    4. For blocks: calculate placement based on the perpendicular to the line connecting block centers
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the input

    # Find connected components of 3s using 4-connectivity
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, component):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 3:
            return
        visited[r][c] = True
        component.append((r, c))
        # Check 4 directions for connectivity
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, component)

    # Find all components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3 and not visited[r][c]:
                component = []
                dfs(r, c, component)
                if component:
                    components.append(sorted(component))

    # Find diagonal pairs (cells that are diagonally adjacent but in different components)
    diagonal_pairs = []
    used_in_pairs = set()

    for i, comp1 in enumerate(components):
        if len(comp1) == 1 and i not in used_in_pairs:
            r1, c1 = comp1[0]
            for j, comp2 in enumerate(components):
                if j > i and len(comp2) == 1 and j not in used_in_pairs:
                    r2, c2 = comp2[0]
                    # Check if diagonally adjacent
                    if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                        diagonal_pairs.append((comp1[0], comp2[0]))
                        used_in_pairs.add(i)
                        used_in_pairs.add(j)
                        break

    # Handle diagonal pairs
    for cell1, cell2 in diagonal_pairs:
        r1, c1 = cell1
        r2, c2 = cell2

        # Center of the pair
        center_r = (r1 + r2) / 2
        center_c = (c1 + c2) / 2

        # Determine direction: '\' diagonal has same sign for dr and dc
        dr, dc = r2 - r1, c2 - c1

        if dr * dc > 0:  # '\' diagonal
            # Perpendicular is '/' diagonal
            offsets = [(-1.5, 1.5), (1.5, -1.5)]
        else:  # '/' diagonal
            # Perpendicular is '\' diagonal
            offsets = [(-1.5, -1.5), (1.5, 1.5)]

        # Place 8s
        for dr_offset, dc_offset in offsets:
            new_r = round(center_r + dr_offset)
            new_c = round(center_c + dc_offset)
            if 0 <= new_r < rows and 0 <= new_c < cols:
                result[new_r][new_c] = 8

    # Handle blocks (components not used in diagonal pairs)
    blocks = [comp for i, comp in enumerate(components) if i not in used_in_pairs and len(comp) > 1]

    if len(blocks) == 2:
        import math

        block1, block2 = blocks

        # Get bounding boxes
        min_r1, max_r1 = min(r for r, c in block1), max(r for r, c in block1)
        min_c1, max_c1 = min(c for r, c in block1), max(c for r, c in block1)
        min_r2, max_r2 = min(r for r, c in block2), max(r for r, c in block2)
        min_c2, max_c2 = min(c for r, c in block2), max(c for r, c in block2)

        height1, width1 = max_r1 - min_r1 + 1, max_c1 - min_c1 + 1
        height2, width2 = max_r2 - min_r2 + 1, max_c2 - min_c2 + 1

        # Calculate centers
        center1_r, center1_c = (min_r1 + max_r1) / 2, (min_c1 + max_c1) / 2
        center2_r, center2_c = (min_r2 + max_r2) / 2, (min_c2 + max_c2) / 2

        # Vector from block1 to block2
        vec_r, vec_c = center2_r - center1_r, center2_c - center1_c

        # Check if blocks form a checkerboard pattern (adjacent or close diagonal)
        # If the bounding boxes form a neat grid, use swap rule
        total_min_r = min(min_r1, min_r2)
        total_max_r = max(max_r1, max_r2)
        total_min_c = min(min_c1, min_c2)
        total_max_c = max(max_c1, max_c2)

        # Check if blocks are aligned and form a 2×2 grid of sub-blocks
        rows_aligned = (max_r1 - min_r1 == max_r2 - min_r2)
        cols_aligned = (max_c1 - min_c1 == max_c2 - min_c2)

        # Check if blocks form a checkerboard (2×2 grid with blocks at diagonal positions)
        # This requires:
        # 1. Blocks are same size
        # 2. Blocks are adjacent (touching)
        # 3. The bounding box is exactly 2x the block size (forms a perfect 2×2 grid)

        total_height = total_max_r - total_min_r + 1
        total_width = total_max_c - total_min_c + 1

        forms_grid = (rows_aligned and cols_aligned and
                      total_height == 2 * height1 and
                      total_width == 2 * width1)

        if forms_grid:
            # Blocks are adjacent and same size - use swap pattern
            # Place 8-block at (block1_rows, block2_cols) and (block2_rows, block1_cols)
            for r in range(min_r1, max_r1 + 1):
                for c in range(min_c2, max_c2 + 1):
                    if 0 <= r < rows and 0 <= c < cols:
                        result[r][c] = 8

            for r in range(min_r2, max_r2 + 1):
                for c in range(min_c1, max_c1 + 1):
                    if 0 <= r < rows and 0 <= c < cols:
                        result[r][c] = 8
        else:
            # Blocks form a parallelogram pattern with two 8-blocks
            # The 4 block centers form a parallelogram with diagonals bisecting at midpoint
            midpoint_r = (center1_r + center2_r) / 2
            midpoint_c = (center1_c + center2_c) / 2

            # Perpendicular to AB vector (90° rotation)
            perp_r, perp_c = -vec_c, vec_r

            # The 8-blocks are placed along the perpendicular through the midpoint
            # Distance is 1.5 times the perpendicular vector (unnormalized)
            for sign in [-1, 1]:
                target_r = midpoint_r + sign * 1.5 * perp_r
                target_c = midpoint_c + sign * 1.5 * perp_c

                # Use same block size as source blocks
                bh, bw = height1, width1

                # Calculate starting position (top-left corner)
                start_r = int(target_r - bh / 2 + 0.5)
                start_c = int(target_c - bw / 2 + 0.5)

                # Place the 8-block
                for dr in range(bh):
                    for dc in range(bw):
                        r, c = start_r + dr, start_c + dc
                        if 0 <= r < rows and 0 <= c < cols:
                            result[r][c] = 8

    return result
