def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a template pattern (connected component with both 1s and 2s)
    2. Input has blocks/cells of 2s not part of template (markers)
    3. Output places scaled template where markers match template's 2-pattern

    Procedure:
    1. Find template and extract relative positions of all cells
    2. Find blocks of isolated 2s
    3. Group blocks that match template's 2-pattern at same scale
    4. For each group, place one scaled template instance
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find template (connected component with both 1s and 2s)
    def find_template():
        visited = set()
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] in [1, 2] and (i, j) not in visited:
                    component = []
                    queue = [(i, j)]
                    visited.add((i, j))
                    while queue:
                        r, c = queue.pop(0)
                        component.append((r, c, grid[r][c]))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < rows and 0 <= nc < cols and
                                grid[nr][nc] in [1, 2] and (nr, nc) not in visited):
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    values = set(v for r, c, v in component)
                    if 1 in values and 2 in values:
                        return component
        return None

    # Find blocks of isolated 2s
    def find_blocks(template):
        template_cells = set((r, c) for r, c, v in template)
        visited = set()
        blocks = []
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2 and (i, j) not in template_cells and (i, j) not in visited:
                    component = []
                    queue = [(i, j)]
                    visited.add((i, j))
                    while queue:
                        r, c = queue.pop(0)
                        component.append((r, c))
                        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < rows and 0 <= nc < cols and
                                grid[nr][nc] == 2 and (nr, nc) not in template_cells and
                                (nr, nc) not in visited):
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    if component:
                        blocks.append(component)
        return blocks

    template = find_template()
    if not template:
        return result

    blocks = find_blocks(template)
    if not blocks:
        return result

    # Get template relative positions
    temp_min_r = min(r for r, c, v in template)
    temp_min_c = min(c for r, c, v in template)
    temp_rel = [(r - temp_min_r, c - temp_min_c, v) for r, c, v in template]
    temp_2s_rel = [(r, c) for r, c, v in temp_rel if v == 2]

    # Determine scale from first block
    b = blocks[0]
    b_height = max(r for r, c in b) - min(r for r, c in b) + 1
    b_width = max(c for r, c in b) - min(c for r, c in b) + 1
    scale = max(b_height, b_width)

    # Group blocks that match the template's 2-pattern
    used = set()

    for i in range(len(blocks)):
        if i in used:
            continue

        # Start a new group with block i
        group_blocks = [blocks[i]]
        used.add(i)

        # If template has multiple 2s, find matching blocks
        if len(temp_2s_rel) > 1:
            # Get position of first block
            b1_min_r = min(r for r, c in blocks[i])
            b1_min_c = min(c for r, c in blocks[i])

            # Try to find blocks matching other 2s in template
            for t_idx in range(1, len(temp_2s_rel)):
                temp_2_r, temp_2_c = temp_2s_rel[t_idx]
                # Expected position for this template 2
                expected_r = b1_min_r + (temp_2_r - temp_2s_rel[0][0]) * scale
                expected_c = b1_min_c + (temp_2_c - temp_2s_rel[0][1]) * scale

                # Find block at this position
                for j in range(len(blocks)):
                    if j in used:
                        continue
                    bj_min_r = min(r for r, c in blocks[j])
                    bj_min_c = min(c for r, c in blocks[j])
                    if bj_min_r == expected_r and bj_min_c == expected_c:
                        group_blocks.append(blocks[j])
                        used.add(j)
                        break

        # Place template for this group
        if temp_2s_rel:
            first_temp_2_r, first_temp_2_c = temp_2s_rel[0]
            first_block_r = min(r for r, c in group_blocks[0])
            first_block_c = min(c for r, c in group_blocks[0])
            offset_r = first_block_r - first_temp_2_r * scale
            offset_c = first_block_c - first_temp_2_c * scale

            # Place scaled template
            for temp_r, temp_c, value in temp_rel:
                out_r = temp_r * scale + offset_r
                out_c = temp_c * scale + offset_c
                for dr in range(scale):
                    for dc in range(scale):
                        nr, nc = out_r + dr, out_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if result[nr][nc] == 0:
                                result[nr][nc] = value

    return result
