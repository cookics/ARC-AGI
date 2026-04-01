def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has one multi-colored template pattern
    2. Input has uniform blocks/cells of colors from the template
    3. Template is scaled by block size and placed to cover those blocks
    4. Multiple placements needed - greedily cover as many targets as possible

    Procedure:
    1. Find multi-colored template
    2. Find all uniform blocks/cells of template colors
    3. Group by scale (block size)
    4. For each scale, try all rotations and positions, greedily place templates
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    all_vals = [grid[i][j] for i in range(rows) for j in range(cols)]
    background = Counter(all_vals).most_common(1)[0][0]

    # Find connected components
    visited = set()
    components = []

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or (r, c) in visited or grid[r][c] == background:
            return []
        visited.add((r, c))
        cells = [(r, c)]
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            cells.extend(dfs(r + dr, c + dc))
        return cells

    for i in range(rows):
        for j in range(cols):
            if (i, j) not in visited and grid[i][j] != background:
                comp = dfs(i, j)
                if comp:
                    components.append(comp)

    if not components:
        return [row[:] for row in grid]

    # Find template (component with most colors)
    template_comp = None
    max_colors = 0
    for comp in components:
        colors = set(grid[r][c] for r, c in comp)
        if len(colors) > max_colors:
            max_colors = len(colors)
            template_comp = comp

    if not template_comp or max_colors <= 1:
        return [row[:] for row in grid]

    # Extract template bounding box
    t_min_r = min(r for r, c in template_comp)
    t_max_r = max(r for r, c in template_comp)
    t_min_c = min(c for r, c in template_comp)
    t_max_c = max(c for r, c in template_comp)

    t_h, t_w = t_max_r - t_min_r + 1, t_max_c - t_min_c + 1
    template = [[background] * t_w for _ in range(t_h)]
    for r in range(t_min_r, t_max_r + 1):
        for c in range(t_min_c, t_max_c + 1):
            template[r - t_min_r][c - t_min_c] = grid[r][c]

    template_colors = set(grid[r][c] for r, c in template_comp)

    result = [row[:] for row in grid]

    # Find uniform blocks/cells (not in template)
    targets = []
    for comp in components:
        if comp == template_comp:
            continue
        colors = set(grid[r][c] for r, c in comp)
        if len(colors) == 1 and list(colors)[0] in template_colors:
            min_r = min(r for r, c in comp)
            max_r = max(r for r, c in comp)
            min_c = min(c for r, c in comp)
            max_c = max(c for r, c in comp)
            h, w = max_r - min_r + 1, max_c - min_c + 1
            color = grid[comp[0][0]][comp[0][1]]
            targets.append((min_r, min_c, h, w, color, comp))

    if not targets:
        return result

    # Group by scale
    from collections import defaultdict
    size_groups = defaultdict(list)
    for idx, (min_r, min_c, h, w, color, comp) in enumerate(targets):
        scale = max(h, w)
        size_groups[scale].append((idx, min_r, min_c, h, w, color, comp))

    # Scaling and rotation
    def scale_template(tmpl, factor):
        th, tw = len(tmpl), len(tmpl[0])
        scaled = [[background] * (tw * factor) for _ in range(th * factor)]
        for tr in range(th):
            for tc in range(tw):
                for dr in range(factor):
                    for dc in range(factor):
                        scaled[tr * factor + dr][tc * factor + dc] = tmpl[tr][tc]
        return scaled

    def rotate_90(g):
        return [[g[len(g) - 1 - c][r] for c in range(len(g))] for r in range(len(g[0]))]

    # Process each scale group
    for scale in sorted(size_groups.keys()):
        group = size_groups[scale]
        scaled_tmpl = scale_template(template, scale)

        # Generate all rotations
        rotations = [scaled_tmpl]
        for _ in range(3):
            rotations.append(rotate_90(rotations[-1]))

        # Greedy placement: keep placing templates until all targets covered
        used = set()

        while len(used) < len(group):
            best_placement = None
            best_covered = set()

            # Try all rotations
            for rot_tmpl in rotations:
                rh, rw = len(rot_tmpl), len(rot_tmpl[0])

                # Build color map for this rotation
                color_map = {}
                for tr in range(rh):
                    for tc in range(rw):
                        if rot_tmpl[tr][tc] != background:
                            if rot_tmpl[tr][tc] not in color_map:
                                color_map[rot_tmpl[tr][tc]] = []
                            color_map[rot_tmpl[tr][tc]].append((tr, tc))

                # Try aligning with each unused target
                for i, (idx, min_r, min_c, h, w, color, comp) in enumerate(group):
                    if i in used:
                        continue

                    if color not in color_map:
                        continue

                    # Try each position of this color in the template
                    for tr, tc in color_map[color]:
                        # Calculate offset to align this template position with target
                        off_r = min_r - tr
                        off_c = min_c - tc

                        # Check bounds
                        valid = True
                        for r in range(rh):
                            for c in range(rw):
                                if rot_tmpl[r][c] != background:
                                    gr, gc = r + off_r, c + off_c
                                    if not (0 <= gr < rows and 0 <= gc < cols):
                                        valid = False
                                        break
                            if not valid:
                                break

                        if not valid:
                            continue

                        # Count how many unused targets this placement covers
                        covered = set()
                        for j, (idx2, min_r2, min_c2, h2, w2, color2, comp2) in enumerate(group):
                            if j in used:
                                continue

                            # Check if all cells of this target are covered correctly
                            all_match = True
                            for (cr, cc) in comp2:
                                tr_check, tc_check = cr - off_r, cc - off_c
                                if not (0 <= tr_check < rh and 0 <= tc_check < rw):
                                    all_match = False
                                    break
                                if rot_tmpl[tr_check][tc_check] != color2:
                                    all_match = False
                                    break

                            if all_match:
                                covered.add(j)

                        # Update best if this covers more targets
                        if len(covered) > len(best_covered):
                            best_covered = covered
                            best_placement = (rot_tmpl, off_r, off_c)

            # Place the best template found
            if best_placement:
                rot_tmpl, off_r, off_c = best_placement
                rh, rw = len(rot_tmpl), len(rot_tmpl[0])
                for r in range(rh):
                    for c in range(rw):
                        gr, gc = r + off_r, c + off_c
                        if 0 <= gr < rows and 0 <= gc < cols:
                            result[gr][gc] = rot_tmpl[r][c]
                used.update(best_covered)
            else:
                # Can't place any more templates
                break

    return result
