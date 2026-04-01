def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing values 0, 1, and 8
    2. Output is a 2D grid where some 1s and 8s are transformed to 2s
    3. The grid contains consecutive groups of identical rows
    4. These groups follow a cyclic transformation pattern: T, N, N (Transform, No-transform, No-transform)
    5. For groups marked 'T': 1s become 2s, 8s become 2s, 0s stay 0s
    6. For groups marked 'N': all values stay unchanged

    Procedure:
    1. Identify consecutive groups of identical rows
    2. Apply T,N,N cyclic pattern to determine which groups to transform
    3. For transform groups: replace 1s and 8s with 2s, keep 0s unchanged
    4. For no-transform groups: keep all values unchanged
    """

    # Identify consecutive groups of identical rows
    groups = []
    i = 0
    while i < len(grid):
        current_row = grid[i]
        group_start = i

        # Find all consecutive identical rows
        while i < len(grid) and grid[i] == current_row:
            i += 1

        groups.append((group_start, i - 1, current_row))

    # Apply T, N, N transformation pattern
    result = [row[:] for row in grid]  # Deep copy

    for group_idx, (start, end, row_pattern) in enumerate(groups):
        # Determine if this group should be transformed using T,N,N cycle
        cycle_position = group_idx % 3
        should_transform = cycle_position == 0  # T, N, N pattern

        if should_transform:
            # Transform 1s and 8s to 2s, keep 0s unchanged
            for row_idx in range(start, end + 1):
                for col_idx in range(len(result[row_idx])):
                    if result[row_idx][col_idx] in [1, 8]:
                        result[row_idx][col_idx] = 2

    return result
