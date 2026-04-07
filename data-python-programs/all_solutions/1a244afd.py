def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing values 1, 6, and 8, output is a modified grid with the same dimensions.
    2. All 6s are removed from the grid (replaced with 8s).
    3. All 1s remain in their original positions unchanged.
    4. New 7s appear at specific positions that are related to the 1s and 6s.
    5. For each 1, there is a closest 6 that shares the same row or column.
    6. The paired 6 is removed and a 7 is placed at the same distance from the 1 but in the opposite direction.
    7. Direction rule: if 6 is above 1 (same column), place 7 to the left of 1.
    8. If 6 is below 1 (same column), place 7 to the right of 1.
    9. If 6 is to the left of 1 (same row), place 7 below 1.
    10. If 6 is to the right of 1 (same row), place 7 above 1.
    11. The distance between 1 and 7 equals the distance between 1 and 6.

    Procedure:
    1. Create a deep copy of the input grid to avoid modifying the original.
    2. Find all positions of 1s and 6s in the grid.
    3. For each 1, find the closest 6 that is in the same row or column.
    4. Remove the paired 6 by replacing it with 8.
    5. Calculate where to place the 7 based on the direction rule and distance.
    6. Place the 7 if the calculated position is within grid bounds.
    7. Remove any remaining unpaired 6s from the grid.
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find all 1s and 6s
    ones = []
    sixes = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                ones.append((r, c))
            elif grid[r][c] == 6:
                sixes.append((r, c))

    # For each 1, find the closest 6 in same row or column
    used_sixes = set()

    for one_r, one_c in ones:
        closest_six = None
        min_distance = float("inf")

        for six_r, six_c in sixes:
            if (six_r, six_c) in used_sixes:
                continue

            # Check if in same row or column
            if one_r == six_r or one_c == six_c:
                distance = abs(one_r - six_r) + abs(one_c - six_c)
                if distance < min_distance:
                    min_distance = distance
                    closest_six = (six_r, six_c)

        if closest_six:
            six_r, six_c = closest_six
            used_sixes.add(closest_six)

            # Remove the 6
            result[six_r][six_c] = 8

            # Calculate where to place the 7
            if one_r == six_r:  # Same row
                if six_c < one_c:  # 6 is to the left of 1
                    # Place 7 below 1
                    seven_r = one_r + (one_c - six_c)
                    seven_c = one_c
                else:  # 6 is to the right of 1
                    # Place 7 above 1
                    seven_r = one_r - (six_c - one_c)
                    seven_c = one_c
            else:  # Same column
                if six_r < one_r:  # 6 is above 1
                    # Place 7 to the left of 1
                    seven_r = one_r
                    seven_c = one_c - (one_r - six_r)
                else:  # 6 is below 1
                    # Place 7 to the right of 1
                    seven_r = one_r
                    seven_c = one_c + (six_r - one_r)

            # Place the 7 if within bounds
            if 0 <= seven_r < rows and 0 <= seven_c < cols:
                result[seven_r][seven_c] = 7

    # Remove any remaining unpaired 6s
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 6:
                result[r][c] = 8

    return result
