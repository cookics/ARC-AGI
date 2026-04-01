def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Single cells adjacent to marker 2 form 3×3 crosses
    2. Rectangular blocks adjacent to markers (1, 2) or other blocks expand into cross shapes
    3. For vertically stacked blocks: horizontal arm at bottom block, vertical arm through both
    4. For horizontally adjacent blocks: vertical arm extended, horizontal arm at original rows
    5. Complex cases like Example 1 use templates from other structures

    Procedure:
    1. Handle single cells with marker 2 → create 3×3 cross patterns
    2. Find rectangular blocks and their adjacent blocks/markers
    3. Expand blocks into crosses based on adjacency orientation
    4. Handle special template-based transformations
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all rectangular blocks first
    visited_block = [[False] * cols for _ in range(rows)]
    blocks = []

    for r in range(rows):
        for c in range(cols):
            if visited_block[r][c] or grid[r][c] == 0:
                continue

            color = grid[r][c]
            comp = []
            queue = [(r, c)]
            visited_block[r][c] = True

            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc))
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited_block[nr][nc] and grid[nr][nc] == color:
                            visited_block[nr][nc] = True
                            queue.append((nr, nc))

            min_r = min(rr for rr, cc in comp)
            max_r = max(rr for rr, cc in comp)
            min_c = min(cc for rr, cc in comp)
            max_c = max(cc for rr, cc in comp)
            height = max_r - min_r + 1
            width = max_c - min_c + 1

            # Check if solid rectangle
            if len(comp) == height * width:
                blocks.append({
                    'color': color,
                    'cells': comp,
                    'min_r': min_r, 'max_r': max_r,
                    'min_c': min_c, 'max_c': max_c,
                    'height': height, 'width': width
                })

    # Find non-rectangular structures (templates like hollow rectangles)
    all_components = []
    visited_all = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if visited_all[r][c] or grid[r][c] == 0:
                continue
            color = grid[r][c]
            comp = []
            queue = [(r, c)]
            visited_all[r][c] = True
            while queue:
                cr, cc = queue.pop(0)
                comp.append((cr, cc))
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited_all[nr][nc] and grid[nr][nc] == color:
                            visited_all[nr][nc] = True
                            queue.append((nr, nc))
            all_components.append((color, comp))

    # Process blocks to find adjacent pairs
    for i, block in enumerate(blocks):
        if block['color'] in [1, 2]:
            continue  # Skip markers

        # Find adjacent blocks
        adjacent = None
        orientation = None

        for j, other in enumerate(blocks):
            if i == j:
                continue

            # Check if vertically stacked
            if (block['min_c'] == other['min_c'] and block['max_c'] == other['max_c']):
                if block['max_r'] + 1 == other['min_r']:  # other is below
                    adjacent = other
                    orientation = 'vertical_below'
                    break
                elif other['max_r'] + 1 == block['min_r']:  # other is above
                    adjacent = other
                    orientation = 'vertical_above'
                    break

            # Check if horizontally adjacent
            if (block['min_r'] == other['min_r'] and block['max_r'] == other['max_r']):
                if block['max_c'] + 1 == other['min_c']:  # other is right
                    adjacent = other
                    orientation = 'horizontal_right'
                    break
                elif other['max_c'] + 1 == block['min_c']:  # other is left
                    adjacent = other
                    orientation = 'horizontal_left'
                    break

        # Special case: vertically stacked blocks not adjacent to markers (Example 1)
        if adjacent and adjacent['color'] not in [1, 2] and orientation == 'vertical_below':
            # This is like Example 1: two blocks stacked, need to use template
            h, w = block['height'], block['width']
            min_r, max_r = block['min_r'], block['max_r']
            min_c, max_c = block['min_c'], block['max_c']

            # Expand horizontally (both directions)
            for rr in range(min_r, max_r + 1):
                for cc in range(max(0, min_c - w), min(cols, max_c + w + 1)):
                    if result[rr][cc] == 0:
                        result[rr][cc] = block['color']

            # Create bars with surrounding color
            for rr in range(adjacent['min_r'], adjacent['max_r'] + 1):
                for cc in range(max(0, min_c - w), min_c):
                    if result[rr][cc] == 0:
                        result[rr][cc] = block['color']
                for cc in range(max_c + 1, min(cols, max_c + w + 1)):
                    if result[rr][cc] == 0:
                        result[rr][cc] = block['color']

            # Find template structure and replicate below
            # Look for interior values in non-rectangular components
            interior_vals = []
            for color, comp in all_components:
                if color == block['color']:
                    continue
                c_min_r = min(rr for rr, cc in comp)
                c_max_r = max(rr for rr, cc in comp)
                c_min_c = min(cc for rr, cc in comp)
                c_max_c = max(cc for rr, cc in comp)
                c_h = c_max_r - c_min_r + 1
                c_w = c_max_c - c_min_c + 1
                # Check if hollow (not a solid rectangle)
                if len(comp) < c_h * c_w:
                    # Find interior values
                    for rr in range(c_min_r + 1, c_max_r):
                        for cc in range(c_min_c + 1, c_max_c):
                            if grid[rr][cc] != 0 and grid[rr][cc] != color:
                                interior_vals.append(grid[rr][cc])

            # Apply template to rows below
            if interior_vals:
                for rr in range(adjacent['max_r'] + 1, min(rows, adjacent['max_r'] + h + 1)):
                    for cc in range(max(0, min_c - w), min_c):
                        if result[rr][cc] == 0:
                            result[rr][cc] = block['color']
                    # Place interior values in middle
                    for cc in range(min_c, max_c + 1):
                        if result[rr][cc] == 0 and interior_vals:
                            val_idx = (rr - (adjacent['max_r'] + 1)) % len(interior_vals)
                            if interior_vals[val_idx] == 0 or result[rr][cc] == 0:
                                result[rr][cc] = interior_vals[val_idx] if interior_vals[val_idx] != adjacent['color'] else 0
                    for cc in range(max_c + 1, min(cols, max_c + w + 1)):
                        if result[rr][cc] == 0:
                            result[rr][cc] = block['color']
            continue

        if adjacent and adjacent['color'] in [1, 2] and len(block['cells']) >= 4:
            # Expand this block into a cross
            h, w = block['height'], block['width']
            min_r, max_r = block['min_r'], block['max_r']
            min_c, max_c = block['min_c'], block['max_c']

            if orientation == 'vertical_below':
                # Adjacent block is below - expand horizontally at adjacent block rows
                # Vertical arm through original position
                for rr in range(min_r, adjacent['max_r'] + h + 1):
                    if rr >= rows:
                        break
                    for cc in range(min_c, max_c + 1):
                        if result[rr][cc] == 0:
                            result[rr][cc] = block['color']

                # Horizontal arm at adjacent block rows
                for rr in range(adjacent['min_r'], adjacent['max_r'] + 1):
                    for cc in range(max(0, min_c - 2 * w), min_c):
                        if result[rr][cc] == 0:
                            result[rr][cc] = block['color']

            elif orientation == 'horizontal_right':
                # Adjacent block is right - expand vertically
                # Vertical arm
                for rr in range(max(0, min_r - h), min(rows, max_r + h + 1)):
                    for cc in range(min_c, max_c + 1):
                        if result[rr][cc] == 0:
                            result[rr][cc] = block['color']

                # Horizontal arm (extend left)
                for rr in range(min_r, max_r + 1):
                    for cc in range(max(0, min_c - w), min_c):
                        if result[rr][cc] == 0:
                            result[rr][cc] = block['color']

    # Handle single cells with marker 2
    processed = set()
    for r in range(rows):
        for c in range(cols):
            if (r, c) in processed or grid[r][c] == 0 or grid[r][c] in [1, 2]:
                continue

            color = grid[r][c]

            # Check if part of a larger block
            is_single = True
            for block in blocks:
                if block['color'] == color and len(block['cells']) > 1:
                    if (r, c) in block['cells']:
                        is_single = False
                        break

            if not is_single:
                continue

            # Check for adjacent marker 2
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 2:
                    # Create 3x3 cross
                    processed.add((r, c))

                    if dc != 0:  # Horizontal marker
                        if dc < 0:  # Marker left, extend right
                            h_cols = [c, c+1, c+2]
                            v_col = c+2
                        else:  # Marker right, extend left
                            h_cols = [c-1, c]
                            v_col = nc

                        for col in h_cols:
                            if 0 <= col < cols and result[r][col] == 0:
                                result[r][col] = color

                        for row_off in [-1, 0, 1]:
                            nr2 = r + row_off
                            if 0 <= nr2 < rows and 0 <= v_col < cols and result[nr2][v_col] == 0:
                                result[nr2][v_col] = color
                    break

    return result
