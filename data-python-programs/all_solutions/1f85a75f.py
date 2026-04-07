def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid containing various integer values scattered throughout.
    2. Most values are distributed sparsely across the grid in random positions.
    3. One specific value forms a compact, dense rectangular cluster in a localized region.
    4. The compact cluster is distinguishable from other values by its tight spatial arrangement.
    5. Output is the extracted rectangular bounding box containing only the compact cluster.
    6. In the output, the cluster value is preserved and empty positions are filled with 0s.
    7. Example patterns: 4s forming 3x3 cluster, 3s forming 5x3 cluster, 2s forming compact region.

    Procedure:
    1. Identify all unique non-zero values in the grid and record their positions.
    2. For each value, calculate its bounding box dimensions and total occurrence count.
    3. Compute compactness ratio as bounding box area divided by occurrence count.
    4. Select the value with the lowest compactness ratio as the target cluster.
    5. Determine the minimal bounding box coordinates that contain all target value positions.
    6. Extract the rectangular region from the bounding box coordinates.
    7. Create output grid where target value positions retain their value and others become 0.
    """

    # Find all unique values and their counts
    value_counts = {}
    for row in grid:
        for val in row:
            value_counts[val] = value_counts.get(val, 0) + 1

    # Find positions of each non-zero value
    value_positions = {}
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            if val != 0:
                if val not in value_positions:
                    value_positions[val] = []
                value_positions[val].append((i, j))

    # Find the value that forms a compact rectangular region
    # This will be the value with positions that form a tight bounding box
    target_value = None
    min_area_ratio = float("inf")

    for val, positions in value_positions.items():
        if len(positions) < 4:  # Skip values with too few occurrences
            continue

        # Find bounding box
        min_row = min(pos[0] for pos in positions)
        max_row = max(pos[0] for pos in positions)
        min_col = min(pos[1] for pos in positions)
        max_col = max(pos[1] for pos in positions)

        # Calculate area ratio (bounding box area vs actual positions)
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        area_ratio = bbox_area / len(positions)

        # The target value should have the most compact representation
        if area_ratio < min_area_ratio:
            min_area_ratio = area_ratio
            target_value = val

    # Extract the rectangular region containing the target value
    if target_value is None:
        return [[0]]

    positions = value_positions[target_value]
    min_row = min(pos[0] for pos in positions)
    max_row = max(pos[0] for pos in positions)
    min_col = min(pos[1] for pos in positions)
    max_col = max(pos[1] for pos in positions)

    # Extract the region
    result = []
    for i in range(min_row, max_row + 1):
        row = []
        for j in range(min_col, max_col + 1):
            if grid[i][j] == target_value:
                row.append(target_value)
            else:
                row.append(0)
        result.append(row)

    return result
