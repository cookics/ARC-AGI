def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with non-zero values forming an incomplete pattern
    2. Output completes the pattern to be both horizontally and vertically symmetric
    3. The transformation applies within the bounding box of all non-zero values
    4. For each cell in the bounding box, we check its 4 symmetric positions (original, horizontal reflection, vertical reflection, both reflections)
    5. If any of the 4 positions has a non-zero value, all 4 positions should have that value

    Procedure:
    1. Create a copy of the input grid
    2. Find the bounding box of all non-zero values (r_min, r_max, c_min, c_max)
    3. For each cell (r, c) in the bounding box:
       - Calculate 4 symmetric positions around the center of bounding box
       - Check all 4 positions in the original grid for any non-zero value
       - Set the result cell to that non-zero value (or 0 if all are 0)
    4. Return the result grid
    """
    import copy

    result = copy.deepcopy(grid)
    rows, cols = len(grid), len(grid[0])

    # Find bounding box of non-zero values
    r_min, r_max = rows, -1
    c_min, c_max = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
                c_min = min(c_min, c)
                c_max = max(c_max, c)

    # If no non-zero values, return original grid
    if r_min > r_max:
        return result

    # For each cell in the bounding box, determine its value based on symmetry
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            # Calculate 4 symmetric positions:
            # 1. Original: (r, c)
            # 2. Horizontal reflection: (r, c_min + c_max - c)
            # 3. Vertical reflection: (r_min + r_max - r, c)
            # 4. Both reflections: (r_min + r_max - r, c_min + c_max - c)
            positions = [
                (r, c),
                (r, c_min + c_max - c),
                (r_min + r_max - r, c),
                (r_min + r_max - r, c_min + c_max - c)
            ]

            # Find any non-zero value from the 4 symmetric positions
            value = 0
            for pr, pc in positions:
                if grid[pr][pc] != 0:
                    value = grid[pr][pc]
                    break

            result[r][c] = value

    return result
