def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid where 8 is background, other values form connected components
    2. Output has same structure but values within each component are reversed
    3. Components are connected by 4-connectivity (up, down, left, right)
    4. The reversal follows a DFS traversal path through each component

    Procedure:
    1. Find all connected components of non-8 values (using 4-connectivity)
    2. For each component, perform DFS to get a consistent path through all nodes
    3. Extract values along this path and reverse them
    4. Assign reversed values back to the original positions in the path
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the grid
    visited = [[False] * cols for _ in range(rows)]

    def dfs(r, c, path, values):
        """Recursive DFS to get path through connected component"""
        visited[r][c] = True
        path.append((r, c))
        values.append(grid[r][c])

        # Explore neighbors in order: up, right, down, left
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and not visited[nr][nc]
                and grid[nr][nc] != 8
            ):
                dfs(nr, nc, path, values)

    # Find all connected components
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8 and not visited[r][c]:
                # Found a new component
                path = []
                values = []
                dfs(r, c, path, values)

                # Reverse the values
                reversed_values = values[::-1]

                # Assign reversed values back to positions
                for i, (pr, pc) in enumerate(path):
                    result[pr][pc] = reversed_values[i]

    return result
