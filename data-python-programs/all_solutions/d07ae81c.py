def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with mostly uniform regions and a few "marker" cells
    2. Markers are cells that differ from the majority (≥3) of their 4 neighbors
    3. Each marker has a special value and an associated background color (the majority neighbor value)
    4. Output has diagonal lines extending from each marker in all 4 diagonal directions
    5. These diagonals paint the marker's value on cells that match any marker's background
    6. Multiple markers' diagonals can cross and interact based on background matching

    Procedure:
    1. Identify all marker cells by checking if they differ from majority of 4 neighbors
    2. Store each marker's position, value, and background color
    3. For each grid position, check if it lies on any diagonal from any marker
    4. If position is on a diagonal and its value matches any marker's background, paint with that marker's value
    5. Return the modified grid
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all special points first
    special_points = []
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current = grid[i][j]
            neighbors = [grid[i - 1][j], grid[i + 1][j], grid[i][j - 1], grid[i][j + 1]]

            # Find the most common neighbor value
            neighbor_counts = {}
            for n in neighbors:
                neighbor_counts[n] = neighbor_counts.get(n, 0) + 1

            background = max(neighbor_counts, key=neighbor_counts.get)
            background_count = neighbor_counts[background]

            if current != background and background_count >= 3:
                special_points.append((i, j, current, background))

    # Apply all diagonal patterns with cross-interaction
    for x in range(rows):
        for y in range(cols):
            # Check if this position is on any diagonal
            for sp_i, sp_j, sp_val, sp_bg in special_points:
                if (x - y == sp_i - sp_j) or (x + y == sp_i + sp_j):
                    # This position is on sp's diagonal
                    # Check if grid value matches any special point's background
                    for (
                        check_sp_i,
                        check_sp_j,
                        check_sp_val,
                        check_sp_bg,
                    ) in special_points:
                        if grid[x][y] == check_sp_bg:
                            result[x][y] = check_sp_val
                            break  # Take the first match

    return result
