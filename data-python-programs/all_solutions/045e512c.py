def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with colored patterns (using 8-connectivity for diagonals)
    2. There are two modes:
       a) Template mode: A template shape exists with indicator lines/cells that replicate it
       b) Self-replication mode: Small patterns complete to staircases and replicate diagonally
    3. Template is the largest 2D shape (size >= 4) that is >2x larger than other patterns
    4. Indicators are simpler patterns (lines or single cells) in different colors
    5. Spacing between replications is 4 units

    Procedure:
    1. Find all connected components (8-connectivity)
    2. Identify the largest 2D template and check if it's significantly larger
    3. Template mode: replicate template based on indicator positions
    4. Self-replication mode:
       - Large patterns (size >= 5) stay in place
       - Small patterns (size < 5) complete to 3x3 staircases and replicate
       - Top-half patterns replicate up-right, bottom-half replicate down-right
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    def complete_diagonal_pattern(component):
        """Complete a small pattern to a diagonal staircase"""
        if not component:
            return []

        min_r = min(r for r, c in component)
        min_c = min(c for r, c in component)

        # Get relative coordinates
        relative = [(r - min_r, c - min_c) for r, c in component]

        # If already 6 cells or more, keep as is
        if len(relative) >= 6:
            return relative

        # Create a 3x3 diagonal staircase: hollow diagonal pattern
        # Base pattern:
        # X X .
        # X . X
        # . X X
        base_pattern = {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1), (2, 2)}

        # Find the shift needed to include the original cells
        # The pattern should be shifted so it encompasses the input
        # For a 2-cell diagonal at (0,0), (1,1), we need to shift up by 1 row
        if len(relative) == 2 and (0, 0) in relative and (1, 1) in relative:
            # Shift pattern up by 1 row
            result = {(r - 1, c) for r, c in base_pattern}
        else:
            # Use base pattern as-is
            result = base_pattern

        return list(result)

    def place_template(
        result, template_shape, base_r, base_c, color, rows, cols, overwrite=True
    ):
        """Place template at given position"""
        for dr, dc in template_shape:
            new_r = base_r + dr
            new_c = base_c + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                if overwrite or result[new_r][new_c] == 0:
                    result[new_r][new_c] = color

    # Find connected components
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, color, component):
        if (
            r < 0
            or r >= rows
            or c < 0
            or c >= cols
            or visited[r][c]
            or grid[r][c] != color
        ):
            return
        visited[r][c] = True
        component.append((r, c))
        # Use 8-connectivity (including diagonals) for staircase patterns
        for dr, dc in [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]:
            dfs(r + dr, c + dc, color, component)

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                component = []
                dfs(i, j, grid[i][j], component)
                components.append((grid[i][j], component))

    # Analyze each component
    analyzed = []
    for color, component in components:
        if not component:
            continue

        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        size = len(component)

        is_2d = height > 1 and width > 1
        is_line = height == 1 or width == 1

        analyzed.append(
            {
                "color": color,
                "component": component,
                "height": height,
                "width": width,
                "size": size,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
                "is_2d": is_2d,
                "is_line": is_line,
            }
        )

    # Find template: largest 2D shape with size >= 4
    # Template mode requires template to be significantly larger than other patterns
    template = None
    max_size = 0

    for item in analyzed:
        if item["is_2d"] and item["size"] >= 4 and item["size"] > max_size:
            max_size = item["size"]
            template = item

    # Check if template is significantly larger (at least 2x) than other patterns
    # Also check if we have indicators (other patterns besides the template)
    has_indicators = False
    is_template_mode = False

    if template:
        # Find the largest non-template pattern size
        max_other_size = 0
        for item in analyzed:
            if item["color"] != template["color"]:
                has_indicators = True
                max_other_size = max(max_other_size, item["size"])

        # Template mode if template is MORE than 2x larger than any other pattern
        if has_indicators and max_size > 2 * max(max_other_size, 1):
            is_template_mode = True

    if is_template_mode:
        # Template mode: replicate template based on indicators
        # Place template
        for r, c in template["component"]:
            result[r][c] = template["color"]

        # Get template shape (relative coordinates)
        template_shape = [
            (r - template["min_r"], c - template["min_c"])
            for r, c in template["component"]
        ]

        # Process indicators
        for item in analyzed:
            if item["color"] == template["color"]:
                continue

            # Vertical line -> replicate horizontally
            if item["height"] > 1 and item["width"] == 1:
                # Check alignment with template
                col_aligned = (
                    abs(item["min_c"] - template["min_c"]) <= template["width"]
                )
                if col_aligned or item["min_c"] > template["max_c"]:
                    # Replicate to the right
                    target_c = item["min_c"]
                    while target_c < cols:
                        place_template(
                            result,
                            template_shape,
                            template["min_r"],
                            target_c,
                            item["color"],
                            rows,
                            cols,
                        )
                        target_c += 4
                else:
                    # Replicate to the left
                    target_c = item["min_c"]
                    while target_c >= 0:
                        place_template(
                            result,
                            template_shape,
                            template["min_r"],
                            target_c,
                            item["color"],
                            rows,
                            cols,
                        )
                        target_c -= 4

            # Horizontal line -> replicate vertically
            elif item["height"] == 1 and item["width"] > 1:
                target_r = item["min_r"]
                while target_r < rows:
                    place_template(
                        result,
                        template_shape,
                        target_r,
                        template["min_c"],
                        item["color"],
                        rows,
                        cols,
                    )
                    target_r += 4

            # Single cell or small pattern -> determine direction
            else:
                template_center_r = (template["min_r"] + template["max_r"]) / 2
                template_center_c = (template["min_c"] + template["max_c"]) / 2
                item_center_r = (item["min_r"] + item["max_r"]) / 2
                item_center_c = (item["min_c"] + item["max_c"]) / 2

                # Check if aligned vertically or horizontally
                if abs(item_center_c - template_center_c) < 1.5:  # Vertical alignment
                    if item_center_r < template_center_r:  # Above
                        target_r = template["min_r"] - 4
                        # Allow negative values since template might partially fit
                        while target_r >= -template["height"]:
                            place_template(
                                result,
                                template_shape,
                                target_r,
                                template["min_c"],
                                item["color"],
                                rows,
                                cols,
                            )
                            target_r -= 4
                    else:  # Below
                        target_r = template["min_r"] + 4
                        while target_r < rows:
                            place_template(
                                result,
                                template_shape,
                                target_r,
                                template["min_c"],
                                item["color"],
                                rows,
                                cols,
                            )
                            target_r += 4
                elif (
                    abs(item_center_r - template_center_r) < 1.5
                ):  # Horizontal alignment
                    if item_center_c < template_center_c:  # Left
                        target_c = template["min_c"] - 4
                        # Allow negative values since template might partially fit
                        while target_c >= -template["width"]:
                            place_template(
                                result,
                                template_shape,
                                template["min_r"],
                                target_c,
                                item["color"],
                                rows,
                                cols,
                            )
                            target_c -= 4
                    else:  # Right
                        target_c = template["min_c"] + 4
                        while target_c < cols:
                            place_template(
                                result,
                                template_shape,
                                template["min_r"],
                                target_c,
                                item["color"],
                                rows,
                                cols,
                            )
                            target_c += 4
    else:
        # Self-replication mode: patterns replicate diagonally
        # First, place large patterns (size >= 5) that don't replicate
        for item in analyzed:
            if item["size"] >= 5:
                for r, c in item["component"]:
                    result[r][c] = item["color"]

        # Then, process small patterns that complete and replicate
        for item in analyzed:
            if item["size"] >= 5:
                continue

            # Small patterns (size < 5) get completed and replicated
            pattern = complete_diagonal_pattern(item["component"])

            # Place original (completed pattern)
            for dr, dc in pattern:
                r, c = item["min_r"] + dr, item["min_c"] + dc
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = item["color"]

            # Determine replication direction based on vertical position
            pattern_center_row = (item["min_r"] + item["max_r"]) / 2

            if pattern_center_row < rows / 2:
                # Top half: replicate up-right only
                mult = 1
                while True:
                    target_r = item["min_r"] - mult * 4
                    target_c = item["min_c"] + mult * 4
                    if target_r < -max(dr for dr, dc in pattern):
                        break
                    place_template(
                        result,
                        pattern,
                        target_r,
                        target_c,
                        item["color"],
                        rows,
                        cols,
                        overwrite=False,
                    )
                    mult += 1
            else:
                # Bottom half: replicate down-right only
                mult = 1
                while True:
                    target_r = item["min_r"] + mult * 4
                    target_c = item["min_c"] + mult * 4
                    # Stop when starting position is out of bounds or far past the edge
                    if target_r >= rows or target_c >= cols:
                        break
                    place_template(
                        result,
                        pattern,
                        target_r,
                        target_c,
                        item["color"],
                        rows,
                        cols,
                        overwrite=False,
                    )
                    mult += 1

    return result
