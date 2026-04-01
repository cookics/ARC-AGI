def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a large grid with a dominant background color and scattered colored regions.
    2. There is a rectangular region filled with 0s that serves as a template.
    3. Within the zero region, certain colors appear at specific positions indicating where they should be placed in output.
    4. The output dimensions match the zero region dimensions.
    5. Each color from the template gets replicated based on the number of large clusters of that color in the entire input grid.

    Procedure:
    1. Identify the background color by finding the most frequent non-zero color.
    2. Find the largest rectangular region of zeros which serves as the template.
    3. Extract the template pattern by identifying which colors appear within the zero region and their positions.
    4. For each color found in the template, scan the entire grid to find all positions of that color.
    5. Group same-colored positions into clusters using proximity-based clustering (distance <= 3).
    6. Count the number of large clusters (size > 1) for each color.
    7. Place each color in the output at template positions, repeated horizontally based on cluster count.
    8. Return the output grid with dimensions matching the zero region template.
    """

    def find_rectangular_region(grid, start_r, start_c, target_color, visited):
        """Find rectangular region of target_color starting from given position"""
        rows, cols = len(grid), len(grid[0])

        if grid[start_r][start_c] != target_color:
            return None

        # Find bounds using flood fill
        min_r = max_r = start_r
        min_c = max_c = start_c
        queue = [(start_r, start_c)]
        region_cells = set()

        while queue:
            r, c = queue.pop(0)
            if (r, c) in region_cells or (r, c) in visited:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != target_color:
                continue

            region_cells.add((r, c))
            visited.add((r, c))
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)

            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((r + dr, c + dc))

        return (min_r, min_c, max_r, max_c)

    rows, cols = len(grid), len(grid[0])

    # Find all colors and their frequencies
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1

    if not color_count:
        return grid

    # Find the background color (most frequent)
    background_color = max(color_count.items(), key=lambda x: x[1])[0]

    # Find the zero region (template)
    zero_regions = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r, c) not in visited:
                region = find_rectangular_region(grid, r, c, 0, visited)
                if region:
                    zero_regions.append(region)

    if not zero_regions:
        return grid

    # Find the largest zero region
    largest_region = max(zero_regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    template_r1, template_c1, template_r2, template_c2 = largest_region

    # Create output grid
    output_height = template_r2 - template_r1 + 1
    output_width = template_c2 - template_c1 + 1
    result = [[0 for _ in range(output_width)] for _ in range(output_height)]

    # Find the template pattern within the zero region
    template_colors = {}
    for r in range(template_r1, template_r2 + 1):
        for c in range(template_c1, template_c2 + 1):
            if grid[r][c] != 0 and grid[r][c] != background_color:
                rel_r = r - template_r1
                rel_c = c - template_c1
                color = grid[r][c]
                if color not in template_colors:
                    template_colors[color] = []
                template_colors[color].append((rel_r, rel_c))

    # For each color found in template, count large clusters and place in output
    for color, template_positions in template_colors.items():
        # Find all positions of this color in the entire grid
        all_positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color:
                    all_positions.append((r, c))

        # Group into clusters
        clusters = []
        used = set()

        for pos in all_positions:
            if pos in used:
                continue

            cluster = [pos]
            used.add(pos)
            queue = [pos]

            while queue:
                cur_r, cur_c = queue.pop(0)

                for other_r, other_c in all_positions:
                    if (other_r, other_c) in used:
                        continue

                    distance = abs(other_r - cur_r) + abs(other_c - cur_c)
                    if distance <= 3:
                        cluster.append((other_r, other_c))
                        used.add((other_r, other_c))
                        queue.append((other_r, other_c))

            clusters.append(cluster)

        # Count large clusters (size > 1)
        large_clusters = [c for c in clusters if len(c) > 1]
        cluster_count = len(large_clusters)

        # Place in output based on template positions and cluster count
        if template_positions and cluster_count > 0:
            base_r, base_c = template_positions[
                0
            ]  # Use first template position as base

            # Place cluster_count instances in consecutive odd columns
            for i in range(cluster_count):
                out_r = base_r
                out_c = base_c + (i * 2)  # Consecutive odd columns (1, 3, 5, ...)

                if 0 <= out_r < output_height and 0 <= out_c < output_width:
                    result[out_r][out_c] = color

    return result
