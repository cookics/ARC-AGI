def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a dominant background color and several rectangular regions of other colors
    2. Output is a smaller grid composed of the colored regions arranged by size
    3. Output height equals the maximum height of the two largest regions
    4. Output width equals the second largest region's width plus 3
    5. Largest region fills the output as base layer
    6. Second largest region is placed on the left side
    7. Smaller regions overlay at the top-left corner in size order

    Procedure:
    1. Find the background color (most frequent value in grid)
    2. Identify all non-background colors and their bounding boxes
    3. Calculate area for each colored region
    4. Sort regions by area in descending order
    5. Determine output dimensions using the two largest regions
    6. Create output grid filled with largest region's color
    7. Overlay second largest region on the left
    8. Overlay remaining smaller regions at top-left positions
    """
    from collections import Counter

    # Find background color (most frequent)
    all_colors = []
    for row in grid:
        all_colors.extend(row)
    color_counts = Counter(all_colors)
    background_color = color_counts.most_common(1)[0][0]

    # Find all non-background regions and their bounding boxes
    regions = {}
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background_color:
                color = grid[r][c]
                if color not in regions:
                    regions[color] = []
                regions[color].append((r, c))

    # For each color, find the tight bounding box
    region_info = {}
    for color, positions in regions.items():
        if not positions:
            continue
        min_r = min(pos[0] for pos in positions)
        max_r = max(pos[0] for pos in positions)
        min_c = min(pos[1] for pos in positions)
        max_c = max(pos[1] for pos in positions)

        region_height = max_r - min_r + 1
        region_width = max_c - min_c + 1

        region_info[color] = {
            "height": region_height,
            "width": region_width,
            "size": region_height * region_width,
        }

    # Sort regions by size (largest first)
    sorted_regions = sorted(
        region_info.items(), key=lambda x: x[1]["size"], reverse=True
    )

    if not sorted_regions:
        return [[background_color]]

    if len(sorted_regions) == 1:
        # Only one region, return it as is
        color, info = sorted_regions[0]
        return [[color for _ in range(info["width"])] for _ in range(info["height"])]

    # Determine output dimensions based on observed pattern
    largest_color, largest_info = sorted_regions[0]
    second_color, second_info = sorted_regions[1]

    # Output height is the max of the two largest regions
    output_height = max(largest_info["height"], second_info["height"])

    # Output width seems to be: second_largest_width + 3 (based on examples)
    # This appears to be a consistent pattern across all examples
    output_width = second_info["width"] + 3

    # Create output grid filled with largest region's color
    result = [
        [largest_color for _ in range(output_width)] for _ in range(output_height)
    ]

    # Place second largest region on the left
    for r in range(min(second_info["height"], output_height)):
        for c in range(min(second_info["width"], output_width)):
            result[r][c] = second_color

    # The largest region fills the remaining space (right side)
    # It's already filled as the base, so no additional work needed

    # Overlay smaller regions at top-left
    for i in range(2, len(sorted_regions)):
        color, info = sorted_regions[i]
        # Overlay at (0,0)
        for r in range(min(info["height"], output_height)):
            for c in range(min(info["width"], output_width)):
                result[r][c] = color

    return result
