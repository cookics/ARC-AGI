def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid where each row's width is divisible by 3
    2. Output is a 2D grid with the same number of rows but shorter width
    3. Each row in the output is the most frequent group when the input row is split into 3 equal parts

    Procedure:
    1. For each row, split it into 3 equal groups
    2. Count the frequency of each group
    3. Return the most frequent group as the output row
    """

    if not grid or not grid[0]:
        return []

    width = len(grid[0])
    group_size = width // 3

    # Verify that width is divisible by 3
    assert width % 3 == 0, f"Width {width} should be divisible by 3"

    result = []

    for row in grid:
        # Split row into 3 groups
        groups = []
        for i in range(3):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            group = row[start_idx:end_idx]
            groups.append(tuple(group))  # Use tuple for hashability

        # Count frequency of each group
        group_counts = {}
        for group in groups:
            group_counts[group] = group_counts.get(group, 0) + 1

        # Find most frequent group
        most_frequent_group = max(group_counts.keys(), key=lambda g: group_counts[g])

        # Convert back to list and add to result
        result.append(list(most_frequent_group))

    return result
