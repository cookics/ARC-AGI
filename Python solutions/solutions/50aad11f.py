def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing connected components of 6s and isolated colored marker numbers (non-0, non-6)
    2. Output is extracted bounding boxes of each component with 6s replaced by their associated marker color
    3. Each connected component of 6s has a nearby colored marker that identifies it
    4. Components are stacked horizontally if markers span more columns, vertically if markers span more rows

    Procedure:
    1. Find all connected components of 6s using flood fill
    2. For each marker (colored number), find the nearest component by Manhattan distance
    3. Extract bounding box for each component, replacing 6s with the marker color
    4. Determine stacking direction based on marker position span (row vs column range)
    5. Sort components by row (vertical stack) or column (horizontal stack) and concatenate
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find all connected components of 6s
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def flood_fill(start_r, start_c):
        """Returns list of (row, col) positions in the component"""
        component = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= cols
                or visited[r][c]
                or grid[r][c] != 6
            ):
                continue

            visited[r][c] = True
            component.append((r, c))

            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))

        return component

    # Find all components of 6s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6 and not visited[r][c]:
                component = flood_fill(r, c)
                if component:
                    components.append(component)

    # For each component, find the nearest colored number and extract the shape
    component_data = []

    for component in components:
        # Find bounding box
        min_r = min(pos[0] for pos in component)
        max_r = max(pos[0] for pos in component)
        min_c = min(pos[1] for pos in component)
        max_c = max(pos[1] for pos in component)

        # Find nearest colored number (not 0 or 6)
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2

        nearest_color = None
        min_distance = float("inf")

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0 and grid[r][c] != 6:
                    distance = abs(r - center_r) + abs(c - center_c)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_color = grid[r][c]

        # Extract component shape
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        shape = [[0] * width for _ in range(height)]

        for r, c in component:
            shape[r - min_r][c - min_c] = nearest_color

        component_data.append(
            {
                "shape": shape,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
                "center_r": center_r,
                "center_c": center_c,
            }
        )

    # Sort components by position for consistent arrangement
    # Determine if arrangement should be horizontal or vertical
    if len(component_data) <= 1:
        if component_data:
            return component_data[0]["shape"]
        else:
            return [[]]

    # Determine arrangement based on component overlaps
    # If components overlap in rows but not columns → arrange horizontally
    # If components overlap in columns but not rows → arrange vertically

    row_overlaps = 0
    col_overlaps = 0
    total_pairs = 0

    for i in range(len(component_data)):
        for j in range(i + 1, len(component_data)):
            comp1 = component_data[i]
            comp2 = component_data[j]
            total_pairs += 1

            # Check row overlap
            row_overlap = not (
                comp1["max_r"] < comp2["min_r"] or comp2["max_r"] < comp1["min_r"]
            )
            if row_overlap:
                row_overlaps += 1

            # Check column overlap
            col_overlap = not (
                comp1["max_c"] < comp2["min_c"] or comp2["max_c"] < comp1["min_c"]
            )
            if col_overlap:
                col_overlaps += 1

    # If more pairs overlap in rows than columns, arrange horizontally
    # If more pairs overlap in columns than rows, arrange vertically
    # If equal or no pairs, use center positions to decide
    if row_overlaps > col_overlaps:
        arrange_horizontally = True
    elif col_overlaps > row_overlaps:
        arrange_horizontally = False
    else:
        # Fall back to center position comparison
        centers_by_row = sorted([comp["center_r"] for comp in component_data])
        centers_by_col = sorted([comp["center_c"] for comp in component_data])

        row_span = (
            centers_by_row[-1] - centers_by_row[0] if len(centers_by_row) > 1 else 0
        )
        col_span = (
            centers_by_col[-1] - centers_by_col[0] if len(centers_by_col) > 1 else 0
        )

        arrange_horizontally = col_span > row_span

    if arrange_horizontally:
        # Sort by column position
        component_data.sort(key=lambda x: x["center_c"])

        # Arrange horizontally
        max_height = max(len(comp["shape"]) for comp in component_data)
        total_width = sum(len(comp["shape"][0]) for comp in component_data)

        result = [[0] * total_width for _ in range(max_height)]

        current_col = 0
        for comp in component_data:
            shape = comp["shape"]
            for r in range(len(shape)):
                for c in range(len(shape[0])):
                    result[r][current_col + c] = shape[r][c]
            current_col += len(shape[0])
    else:
        # Sort by row position
        component_data.sort(key=lambda x: x["center_r"])

        # Arrange vertically
        max_width = max(len(comp["shape"][0]) for comp in component_data)
        total_height = sum(len(comp["shape"]) for comp in component_data)

        result = [[0] * max_width for _ in range(total_height)]

        current_row = 0
        for comp in component_data:
            shape = comp["shape"]
            for r in range(len(shape)):
                for c in range(len(shape[0])):
                    result[current_row + r][c] = shape[r][c]
            current_row += len(shape)

    return result
