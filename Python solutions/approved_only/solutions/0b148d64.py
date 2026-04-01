def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing different colored regions separated by zeros.
    2. The grid contains exactly two distinct non-zero values representing different colored regions.
    3. Output is a rectangular subgrid containing one of the colored regions.
    4. The pattern shows that we need to extract the region with smaller bounding box area.
    5. Each colored region forms a connected rectangular area within the grid.

    Procedure:
    1. Find all unique non-zero values in the grid.
    2. For each unique value, calculate its bounding box coordinates.
    3. Compare the areas of the bounding boxes for each colored region.
    4. Select the region with the smaller bounding box area.
    5. Extract and return the rectangular subgrid containing that region.
    """

    rows, cols = len(grid), len(grid[0])

    # Find all unique non-zero values in the grid
    unique_values = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                unique_values.add(grid[r][c])

    # For each unique value, find its bounding box
    value_regions = {}
    for value in unique_values:
        min_r, max_r = rows, -1
        min_c, max_c = cols, -1

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == value:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        if max_r >= 0:  # Found at least one cell with this value
            value_regions[value] = (min_r, max_r, min_c, max_c)

    # Now I need to determine which region to extract
    # Let me check the pattern by looking at the area of each region
    if len(value_regions) == 2:
        # Find the region with smaller area
        region_areas = []
        for value, (min_r, max_r, min_c, max_c) in value_regions.items():
            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            region_areas.append((area, value, min_r, max_r, min_c, max_c))

        # Sort by area and pick the smaller one
        region_areas.sort()
        _, target_value, min_r, max_r, min_c, max_c = region_areas[0]

        # Extract the region
        result = []
        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(grid[r][c])
            result.append(row)

        return result

    return [[]]
