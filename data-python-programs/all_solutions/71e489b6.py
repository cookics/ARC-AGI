def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with values 0 and 1
    2. Output modifies cells based on their local 8-neighborhood:
       - 0-cells with >50% neighbors as 1s get surrounded by 7s
       - 1-cells with >50% neighbors as 0s get changed to 0
    3. Process each cell individually based on its immediate neighbors

    Procedure:
    1. First pass: change 1-cells with >50% 0-neighbors to 0
    2. Second pass: for 0-cells with >50% 1-neighbors, place 7s at ALL 8 neighbors
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # First pass: change isolated 1s to 0s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                neighbor_count = 0
                zero_count = 0

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor_count += 1
                            if grid[nr][nc] == 0:
                                zero_count += 1

                # Use stricter threshold (> 60%) to avoid changing edge cells
                if neighbor_count > 0 and zero_count > neighbor_count * 0.6:
                    result[r][c] = 0

    # Second pass: identify all 0-cells that need borders
    cells_needing_borders = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                neighbor_count = 0
                one_count = 0

                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbor_count += 1
                            if grid[nr][nc] == 1:
                                one_count += 1

                if neighbor_count > 0 and one_count > neighbor_count / 2:
                    cells_needing_borders.add((r, c))

    # Third pass: place 7s at neighbors (but not at other 0-cells that need borders)
    for r, c in cells_needing_borders:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Don't place 7 on other 0-cells that need borders
                    if (nr, nc) not in cells_needing_borders:
                        result[nr][nc] = 7

    return result
