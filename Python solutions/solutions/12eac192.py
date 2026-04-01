def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing integers including 0 and other positive values.
    2. Output is a 2D grid with the same dimensions as input.
    3. Background cells with value 0 always remain unchanged.
    4. Non-zero values form connected components using 4-connectivity (up, down, left, right).
    5. Connected components of non-zero values with size >= 3 stay unchanged.
    6. Connected components of non-zero values with size < 3 are replaced with value 3.

    Procedure:
    1. Create a deep copy of the input grid to store results.
    2. Initialize a visited matrix to track processed cells.
    3. Iterate through each cell in the grid.
    4. For each unvisited non-zero cell, use flood fill to find its connected component.
    5. If the connected component size is less than 3, replace all cells in it with value 3.
    6. Return the modified grid.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy
    visited = [[False] * cols for _ in range(rows)]

    def flood_fill(start_r, start_c, target_value):
        """Find connected component and return list of coordinates"""
        if visited[start_r][start_c] or grid[start_r][start_c] != target_value:
            return []

        component = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r][c] or grid[r][c] != target_value:
                continue

            visited[r][c] = True
            component.append((r, c))

            # Add 4-connected neighbors
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

        return component

    # Process each cell
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and grid[i][j] != 0:
                component = flood_fill(i, j, grid[i][j])

                # If component size < 3, replace with 3
                if len(component) < 3:
                    for r, c in component:
                        result[r][c] = 3

    return result
