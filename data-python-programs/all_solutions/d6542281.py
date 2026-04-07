def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a background value and connected components (patterns)
    2. Large patterns are "templates" that should be copied
    3. Small patterns or isolated cells are "markers" indicating where to copy
    4. A template is copied when isolated markers match some anchor values in the template
    5. The copying aligns anchor values in template with corresponding marker values

    Procedure:
    1. Find background value (most frequent)
    2. Find all connected components
    3. Separate into templates (large) and markers (small)
    4. For each template, find all markers that share values with it
    5. Try to find offset that aligns template anchor values with marker values
    6. Copy template to new position if valid alignment found
    """
    from collections import Counter, deque

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find background value (most frequent)
    all_values = [grid[r][c] for r in range(rows) for c in range(cols)]
    background = Counter(all_values).most_common(1)[0][0]

    # Find connected components using BFS
    def get_component(start_r, start_c, visited):
        component = []
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))

        while queue:
            r, c = queue.popleft()
            component.append((r, c, grid[r][c]))

            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] != background:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return component

    visited = set()
    components = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and (r, c) not in visited:
                comp = get_component(r, c, visited)
                components.append(comp)

    if not components:
        return result

    # Sort by size
    components.sort(key=len, reverse=True)

    # Separate templates and markers
    # Templates are larger components (> 3 cells), markers are smaller
    if len(components) == 0:
        return result

    # Identify threshold - components significantly smaller than largest are markers
    if components:
        max_size = len(components[0])
        templates = [c for c in components if len(c) > 3]
        markers = [c for c in components if len(c) <= 3]
    else:
        return result

    # Track all used (template_id, offset) pairs globally to avoid duplicates
    used_copies = set()

    # For each template, try to align it with markers
    for template_idx, template in enumerate(templates):
        # Get template cells organized by value
        template_by_value = {}
        for r, c, v in template:
            if v not in template_by_value:
                template_by_value[v] = []
            template_by_value[v].append((r, c))

        # For each marker, try to align this template
        for marker in markers:
            # Get marker cells by value
            marker_by_value = {}
            for r, c, v in marker:
                if v not in marker_by_value:
                    marker_by_value[v] = []
                marker_by_value[v].append((r, c))

            # Find shared values
            shared_values = set(template_by_value.keys()) & set(marker_by_value.keys())
            if not shared_values:
                continue

            # Try to find offset that aligns this marker with template
            first_value = list(shared_values)[0]

            for t_pos in template_by_value[first_value]:
                for m_pos in marker_by_value[first_value]:
                    offset_r = m_pos[0] - t_pos[0]
                    offset_c = m_pos[1] - t_pos[1]

                    # Check if ALL marker cells align with template
                    all_match = True
                    for val in shared_values:
                        for mr, mc in marker_by_value[val]:
                            template_r = mr - offset_r
                            template_c = mc - offset_c
                            if (template_r, template_c) not in template_by_value[val]:
                                all_match = False
                                break
                        if not all_match:
                            break

                    # If valid and not already copied, copy template
                    if all_match:
                        copy_key = (template_idx, offset_r, offset_c)
                        if copy_key not in used_copies:
                            used_copies.add(copy_key)

                            for t_r, t_c, t_v in template:
                                new_r = t_r + offset_r
                                new_c = t_c + offset_c
                                if 0 <= new_r < rows and 0 <= new_c < cols:
                                    result[new_r][new_c] = t_v

                        break

                if all_match:
                    break

    return result
