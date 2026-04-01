def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background value 8 and nested rectangular frames of different values
    2. Output is a sorted list of unique non-8 values, each as a single-element list
    3. Primary ordering is by leftmost column position where each value first appears
    4. Alternative ordering by bounding box area (descending) when nesting patterns differ

    Procedure:
    1. Find all positions of each unique non-8 value
    2. Calculate leftmost column and bounding box area for each value
    3. Determine ordering strategy based on spatial relationships
    4. Return sorted values as single-element lists
    """

    # Find all positions for each unique non-8 value
    value_positions = {}
    rows = len(grid)
    cols = len(grid[0])

    for row in range(rows):
        for col in range(cols):
            val = grid[row][col]
            if val != 8:
                if val not in value_positions:
                    value_positions[val] = []
                value_positions[val].append((row, col))

    # Calculate metrics for each value
    metrics = {}
    for val, positions in value_positions.items():
        leftmost_col = min(pos[1] for pos in positions)
        rightmost_col = max(pos[1] for pos in positions)
        topmost_row = min(pos[0] for pos in positions)
        bottommost_row = max(pos[0] for pos in positions)

        area = (bottommost_row - topmost_row + 1) * (rightmost_col - leftmost_col + 1)

        metrics[val] = {"leftmost_col": leftmost_col, "area": area}

    # Calculate both potential orderings
    leftmost_sorted = sorted(metrics.keys(), key=lambda v: metrics[v]["leftmost_col"])
    area_sorted = sorted(metrics.keys(), key=lambda v: -metrics[v]["area"])

    # Determine which ordering to use
    # Use area-based ordering when the leftmost value has disproportionately small area
    use_area_sorting = False

    if len(leftmost_sorted) > 1 and leftmost_sorted != area_sorted:
        first_val = leftmost_sorted[0]
        first_area = metrics[first_val]["area"]

        # Count how many values have significantly larger area
        larger_count = sum(1 for v in metrics if metrics[v]["area"] > first_area * 2)

        # Use area sorting if multiple values are much larger
        if larger_count >= 2:
            use_area_sorting = True

    # Select final ordering
    if use_area_sorting:
        sorted_values = area_sorted
    else:
        sorted_values = leftmost_sorted

    # Return each value as a single-element list
    result = [[val] for val in sorted_values]
    return result
