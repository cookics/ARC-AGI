def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing 0s and 8s.
    2. The output is the same grid with some 0s changed to 1s.
    3. A 0 becomes 1 when it is surrounded by 8s in an L-shaped corner pattern.
    4. The L-shaped pattern requires 8s in exactly 3 adjacent positions that form a right angle.
    5. There are 4 possible L-patterns: top-left, top-right, bottom-left, and bottom-right corners.
    6. Each L-pattern consists of 3 specific adjacent directions that create a corner shape.

    Procedure:
    1. Create a copy of the input grid to store the result.
    2. Iterate through each cell in the grid.
    3. For each cell that contains 0, find all adjacent cells containing 8.
    4. Check if the adjacent 8s form any of the 4 possible L-shaped corner patterns.
    5. If an L-pattern is found, change the 0 to 1 in the result grid.
    6. Return the modified grid.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Copy the grid

    # Direction vectors for 8 adjacent cells (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:  # Only check empty cells
                # Get all adjacent 8s
                adjacent_8s = []
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 8:
                        adjacent_8s.append((dr, dc))

                # Check if these 8s form an L-shape or corner pattern
                if len(adjacent_8s) >= 3:
                    # Check for L-shaped patterns
                    # An L-shape has 3 consecutive directions forming a right angle
                    direction_set = set(adjacent_8s)

                    # Check all possible L-shapes (corner patterns)
                    l_patterns = [
                        [(-1, -1), (-1, 0), (0, -1)],  # top-left corner
                        [(-1, 0), (-1, 1), (0, 1)],  # top-right corner
                        [(0, -1), (1, -1), (1, 0)],  # bottom-left corner
                        [(0, 1), (1, 0), (1, 1)],  # bottom-right corner
                    ]

                    for pattern in l_patterns:
                        if all(direction in direction_set for direction in pattern):
                            result[r][c] = 1
                            break

    return result
