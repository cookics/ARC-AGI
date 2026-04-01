def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains scattered rectangular frames/regions of different colors.
    2. Output is a compact grid with concentric rectangular layers.
    3. Each unique color forms one layer in the output.
    4. Colors whose bounding boxes contain other colors' pixels go in outer layers.
    5. This creates a nesting order based on spatial containment relationships.
    6. Output size is 2*num_colors - 1.

    Procedure:
    1. Extract all unique non-zero colors and their positions.
    2. For each color, determine its bounding box.
    3. Count how many OTHER colors have pixels inside this color's bounding box.
    4. Sort colors by containment count (more contains = outer layer).
    5. Create concentric rectangular output with sorted colors.
    """

    # Extract all unique non-zero colors
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)

    colors = sorted(list(colors))

    # Calculate bounding box and positions for each color
    color_info = {}
    for color in colors:
        positions = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == color:
                    positions.append((i, j))

        if positions:
            min_r = min(pos[0] for pos in positions)
            max_r = max(pos[0] for pos in positions)
            min_c = min(pos[1] for pos in positions)
            max_c = max(pos[1] for pos in positions)

            color_info[color] = {
                "positions": positions,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
            }

    # Compute distance of bbox center from grid center
    rows = len(grid)
    cols = len(grid[0])
    grid_center_r = (rows - 1) / 2.0
    grid_center_c = (cols - 1) / 2.0

    bbox_center_dist = {}
    for color in colors:
        info = color_info[color]
        bbox_center_r = (info["min_r"] + info["max_r"]) / 2.0
        bbox_center_c = (info["min_c"] + info["max_c"]) / 2.0
        dist = ((bbox_center_r - grid_center_r) ** 2 + (bbox_center_c - grid_center_c) ** 2) ** 0.5
        bbox_center_dist[color] = dist

    # Sort colors: LARGER distance from grid center = outer layer, then by color value
    sorted_colors = sorted(
        colors,
        key=lambda c: (-bbox_center_dist[c], c),
    )

    # Create output with concentric rectangular layers
    num_colors = len(colors)
    output_size = 2 * num_colors - 1

    result = [[0 for _ in range(output_size)] for _ in range(output_size)]

    for layer, color in enumerate(sorted_colors):
        # Fill the rectangular border for this layer
        for i in range(layer, output_size - layer):
            for j in range(layer, output_size - layer):
                # Fill border of current layer (or fill completely if innermost)
                if (
                    layer == num_colors - 1  # Innermost layer
                    or i == layer  # Top edge
                    or i == output_size - layer - 1  # Bottom edge
                    or j == layer  # Left edge
                    or j == output_size - layer - 1  # Right edge
                ):
                    result[i][j] = color

    return result
