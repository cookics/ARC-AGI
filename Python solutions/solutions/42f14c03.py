def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 16x16 grid with a dominant background color and several colored patterns
    2. Output is a smaller grid extracted from one main "hollow" pattern's bounding box
    3. One pattern forms a hollow/frame structure (color on borders, background inside)
    4. Other patterns are small solid blocks that serve as "filler" colors
    5. The hollow interior positions are filled with colors from the filler patterns
    6. When multiple fillers exist, they are assigned spatially based on column position

    Procedure:
    1. Identify the background color (most frequent value in grid)
    2. Find all positions for each non-background color
    3. Identify frame patterns (colors where bounding box area > cell count)
    4. Select the main frame pattern (first/largest frame)
    5. Extract the frame to its minimal bounding box
    6. Identify small filler objects (non-frame colored patterns)
    7. Replace background cells in the extracted frame with filler colors
    8. If multiple fillers, assign based on horizontal position (left filler on left, right filler on right)
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Find background color (most common)
    all_colors = []
    for row in grid:
        all_colors.extend(row)
    background = Counter(all_colors).most_common(1)[0][0]

    # Find positions of each color
    color_positions = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color not in color_positions:
                color_positions[color] = []
            color_positions[color].append((r, c))

    # Find rectangular frame patterns
    def find_frame_patterns():
        patterns = []

        for color, positions in color_positions.items():
            if color == background or len(positions) < 8:
                continue

            # Get bounding rectangle
            min_r = min(r for r, c in positions)
            max_r = max(r for r, c in positions)
            min_c = min(c for r, c in positions)
            max_c = max(c for r, c in positions)

            # Extract rectangle
            rect = []
            for r in range(min_r, max_r + 1):
                row_data = []
                for c in range(min_c, max_c + 1):
                    row_data.append(grid[r][c])
                rect.append(row_data)

            rect_rows, rect_cols = len(rect), len(rect[0])

            # Check if has frame structure (background cells inside)
            has_interior_background = False
            background_positions = []

            for r in range(rect_rows):
                for c in range(rect_cols):
                    if rect[r][c] == background:
                        # Check if it's truly interior (not on border)
                        if 0 < r < rect_rows - 1 or 0 < c < rect_cols - 1:
                            has_interior_background = True
                            background_positions.append((r, c))

            if has_interior_background:
                patterns.append(
                    {
                        "rect": rect,
                        "color": color,
                        "position": (min_r, min_c),
                        "background_positions": background_positions,
                        "size": rect_rows * rect_cols,
                    }
                )

        return patterns

    # Find small colored objects for replacement
    def find_small_objects():
        small_objects = []

        for color, positions in color_positions.items():
            if color == background:
                continue

            # Check if it's a small object (not part of large frame)
            if len(positions) <= 6:
                avg_col = sum(c for r, c in positions) / len(positions)
                small_objects.append(
                    {"color": color, "avg_col": avg_col, "positions": positions}
                )

        return small_objects

    patterns = find_frame_patterns()
    small_objects = find_small_objects()

    if not patterns:
        return grid

    # Choose template (first pattern or most suitable)
    template = patterns[0]
    result = [row[:] for row in template["rect"]]
    template_color = template["color"]

    # Filter replacement colors (exclude template color)
    replacement_objects = [
        obj for obj in small_objects if obj["color"] != template_color
    ]

    if not replacement_objects:
        return result

    # Apply replacement strategy
    result_rows, result_cols = len(result), len(result[0])

    if len(replacement_objects) >= 2:
        # Spatial replacement based on column position
        replacement_objects.sort(key=lambda x: x["avg_col"])
        left_color = replacement_objects[0]["color"]
        right_color = replacement_objects[-1]["color"]

        # Replace background cells spatially
        for r in range(result_rows):
            for c in range(result_cols):
                if result[r][c] == background:
                    if c < result_cols / 2:
                        result[r][c] = left_color
                    else:
                        result[r][c] = right_color
    else:
        # Uniform replacement
        replacement_color = replacement_objects[0]["color"]
        for r in range(result_rows):
            for c in range(result_cols):
                if result[r][c] == background:
                    result[r][c] = replacement_color

    return result
