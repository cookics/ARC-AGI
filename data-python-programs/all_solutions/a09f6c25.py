def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains background and foreground cells (value 2)
    2. Grid may be divided into sections by sparse rows (rows with ≤1 cell of value 2)
    3. Within each section:
       - Largest component by size gets color (background - 1)
       - Components with width >= 6 get color 1
       - Other components get color 3
    4. Small components (size < 3) are removed

    Procedure:
    1. Find background color and identify sections
    2. Find connected components
    3. For each section, identify largest component
    4. Assign colors based on size and width
    """
    from collections import deque

    if not grid or not grid[0]:
        return grid

    rows = len(grid)
    cols = len(grid[0])

    # Find background color
    color_count = {}
    for row in grid:
        for val in row:
            if val != 2:
                color_count[val] = color_count.get(val, 0) + 1
    background = max(color_count.items(), key=lambda x: x[1])[0] if color_count else 0

    # Identify sections (separated by rows with ≤1 foreground cell)
    sections = []
    current_start = 0
    for r in range(rows):
        count = sum(1 for c in range(cols) if grid[r][c] == 2)
        if count <= 1:
            if r > current_start:
                sections.append((current_start, r))
            current_start = r + 1
    if current_start < rows:
        sections.append((current_start, rows))

    if not sections:
        sections = [(0, rows)]

    # Find connected components
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def bfs(start_r, start_c):
        component = []
        queue = deque([(start_r, start_c)])
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.popleft()
            component.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == 2:
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        return component

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2 and not visited[r][c]:
                component = bfs(r, c)
                components.append(component)

    # Group components by section
    section_components = [[] for _ in sections]
    for comp in components:
        avg_row = sum(r for r, c in comp) // len(comp)
        for idx, (start, end) in enumerate(sections):
            if start <= avg_row < end:
                section_components[idx].append(comp)
                break

    # Initialize result
    result = [row[:] for row in grid]

    # Process each section
    for section_idx, comps in enumerate(section_components):
        # Find largest component in this section
        largest_comp = None
        max_size = 0
        for comp in comps:
            if len(comp) > max_size:
                max_size = len(comp)
                largest_comp = comp

        # Assign colors
        for comp in comps:
            if len(comp) < 3:
                for r, c in comp:
                    result[r][c] = background
                continue

            # Calculate bounding box width
            min_c = min(c for r, c in comp)
            max_c = max(c for r, c in comp)
            width = max_c - min_c + 1

            # Determine color
            if comp is largest_comp:
                color = background - 1
            elif width >= 6:
                color = 1
            else:
                color = 3

            for r, c in comp:
                result[r][c] = color

    return result
