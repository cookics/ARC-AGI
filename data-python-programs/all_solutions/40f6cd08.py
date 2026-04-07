def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains multiple rectangular regions filled with values
    2. One rectangle is a "template" with a multi-layered pattern
    3. Other rectangles are uniformly filled
    4. Task: Apply the template pattern to uniform rectangles, scaling as needed

    Procedure:
    1. Find all rectangular connected components
    2. Identify template (has most unique colors)
    3. For each uniform rectangle, scale the template pattern to fit it
    """

    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all rectangular connected components
    rectangles = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r][c] != 0:
                # BFS to find connected component
                stack = [(r, c)]
                component = []

                while stack:
                    cr, cc = stack.pop()
                    if (
                        (cr, cc) in visited
                        or cr < 0
                        or cr >= rows
                        or cc < 0
                        or cc >= cols
                        or grid[cr][cc] == 0
                    ):
                        continue

                    visited.add((cr, cc))
                    component.append((cr, cc))

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        stack.append((cr + dr, cc + dc))

                if component:
                    min_r = min(r for r, c in component)
                    max_r = max(r for r, c in component)
                    min_c = min(c for r, c in component)
                    max_c = max(c for r, c in component)

                    # Check if it's a perfect rectangle
                    expected_size = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if len(component) == expected_size:
                        # Extract pattern
                        pattern = []
                        for rr in range(min_r, max_r + 1):
                            row = []
                            for cc in range(min_c, max_c + 1):
                                row.append(grid[rr][cc])
                            pattern.append(row)

                        unique_colors = len(set(val for row in pattern for val in row))

                        rectangles.append(
                            {
                                "bounds": (min_r, min_c, max_r, max_c),
                                "pattern": pattern,
                                "unique_colors": unique_colors,
                            }
                        )

    if len(rectangles) < 2:
        return result

    # Find template (most unique colors)
    template = max(rectangles, key=lambda x: x["unique_colors"])
    template_pattern = template["pattern"]
    src_h, src_w = len(template_pattern), len(template_pattern[0])

    # Apply template to other rectangles
    for rect in rectangles:
        if rect == template:
            continue

        min_r, min_c, max_r, max_c = rect["bounds"]
        tgt_h, tgt_w = max_r - min_r + 1, max_c - min_c + 1

        # Scale template pattern to target size
        for i in range(tgt_h):
            for j in range(tgt_w):
                # Map using pixel center coordinates with floor
                src_i = int(i * src_h / tgt_h)
                src_j = int(j * src_w / tgt_w)

                # Clamp to valid range
                src_i = min(src_i, src_h - 1)
                src_j = min(src_j, src_w - 1)

                result[min_r + i][min_c + j] = template_pattern[src_i][src_j]

    return result
