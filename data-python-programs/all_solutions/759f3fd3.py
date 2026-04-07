def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains a cross pattern made of 3s (one horizontal line, one vertical line)
    2. The cross divides the grid into 4 quadrants
    3. The pattern is applied to a conceptual grid without the cross
    4. The filling uses bitwise formula: fill if (r | c) % 3 != 1 in conceptual coords
    5. Both horizontal and vertical mirroring exist around the center of conceptual grid

    Procedure:
    1. Find the cross lines (horizontal row and vertical column of 3s)
    2. For each cell not on the cross:
       a. Map to conceptual grid (removing cross)
       b. Apply mirroring if in right/bottom halves
       c. Apply bitwise filling formula
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find the cross lines
    h_line = -1  # horizontal line row
    v_line = -1  # vertical line column

    # Find horizontal line of 3s
    for r in range(rows):
        if all(grid[r][c] == 3 for c in range(cols)):
            h_line = r
            break

    # Find vertical line of 3s
    for c in range(cols):
        if all(grid[r][c] == 3 for r in range(rows)):
            v_line = c
            break

    if h_line == -1 or v_line == -1:
        return result

    # Create conceptual grid pattern (without cross)
    concept_rows = rows - 1
    concept_cols = cols - 1

    # Fill pattern in conceptual grid
    concept_grid = [[0] * concept_cols for _ in range(concept_rows)]

    for r in range(concept_rows):
        for c in range(concept_cols):
            # Apply bitwise pattern: fill if (r | c) % 3 != 1
            if (r | c) % 3 != 1:
                concept_grid[r][c] = 4

    # Map conceptual grid back to result grid (inserting cross)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:  # Skip cross
                continue

            # Map to conceptual coordinates
            concept_r = r if r < h_line else r - 1
            concept_c = c if c < v_line else c - 1

            result[r][c] = concept_grid[concept_r][concept_c]

    return result
