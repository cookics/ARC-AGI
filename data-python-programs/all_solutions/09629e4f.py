def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is 11x11 grid with separators (5s) at rows/cols 3,7 creating 9 regions
    2. Output fills each 3x3 region with single value from {0,2,3,4,6}
    3. Exactly 4 regions get values {2,3,4,6}, others get 0
    4. Use greedy matching: For each value, find region with most occurrences

    Procedure:
    1. Count occurrences of each value in each region
    2. Greedily assign values to regions based on highest count
    3. Each value/region used once
    """
    from collections import Counter

    result = [[0] * 11 for _ in range(11)]

    # Set separators
    for i in range(11):
        result[3][i] = result[7][i] = 5
        result[i][3] = result[i][7] = 5

    target_values = [2, 3, 4, 6]

    # Count occurrences of each value in each region
    region_counts = {}
    for reg_r in range(3):
        for reg_c in range(3):
            row_start = reg_r * 4
            col_start = reg_c * 4
            counts = Counter()
            for dr in range(3):
                for dc in range(3):
                    val = grid[row_start + dr][col_start + dc]
                    if val in target_values:
                        counts[val] += 1
            region_counts[(reg_r, reg_c)] = counts

    # Greedy bipartite matching
    # Create list of (count, val, region) and sort descending
    candidates = []
    for (reg_r, reg_c), counts in region_counts.items():
        for val, count in counts.items():
            if count > 0:
                candidates.append((count, val, (reg_r, reg_c)))

    candidates.sort(reverse=True)

    assignment = {}
    used_values = set()
    used_regions = set()

    for count, val, (reg_r, reg_c) in candidates:
        if val not in used_values and (reg_r, reg_c) not in used_regions:
            assignment[(reg_r, reg_c)] = val
            used_values.add(val)
            used_regions.add((reg_r, reg_c))

    # Fill regions
    for reg_r in range(3):
        for reg_c in range(3):
            fill_val = assignment.get((reg_r, reg_c), 0)
            row_start = reg_r * 4
            col_start = reg_c * 4
            for dr in range(3):
                for dc in range(3):
                    result[row_start + dr][col_start + dc] = fill_val

    # Restore separators
    for i in range(11):
        result[3][i] = result[7][i] = 5
        result[i][3] = result[i][7] = 5

    return result
