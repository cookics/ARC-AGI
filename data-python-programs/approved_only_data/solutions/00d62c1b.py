def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 0s (empty cells) and 3s (boundary cells)
    2. Output fills enclosed 0s (surrounded by 3s) with 4s

    Procedure:
    1. Use flood fill from edges to mark all 0s reachable from boundary
    2. Replace unmarked 0s with 4s
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 0:
            return
        visited[r][c] = True
        # Check all 4 directions
        flood_fill(r + 1, c)
        flood_fill(r - 1, c)
        flood_fill(r, c + 1)
        flood_fill(r, c - 1)

    # Flood fill from all edge cells
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and grid[i][j] == 0:
                flood_fill(i, j)

    # Replace unvisited 0s with 4s
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0 and not visited[i][j]:
                result[i][j] = 4

    return result
