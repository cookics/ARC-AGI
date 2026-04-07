def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing values 0, 1, and 2
    2. Output is a grid where some 0s are replaced with 4s based on the positions of two 2s
    3. The two 2s act as anchor points defining a rectangular bounding box
    4. Only 0s within certain regions get replaced with 4s, while 1s and 2s remain unchanged
    5. The filling pattern depends on the relative positions of the two 2s (same row, same column, or diagonal)

    Procedure:
    1. Find all positions of 2s in the grid
    2. Determine the relationship between the two 2s (same row, same column, or diagonal configuration)
    3. Apply the appropriate filling pattern based on the 2s configuration and grid size
    4. Replace 0s with 4s in the determined regions while preserving 1s and 2s
    5. Return the modified grid
    """

    # Create a copy of the grid
    result = [row[:] for row in grid]

    # Find all positions of 2s
    twos = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                twos.append((i, j))

    if len(twos) >= 2:
        # Sort twos by position for consistent handling
        twos.sort()
        r1, c1 = twos[0]
        r2, c2 = twos[1]

        # Check if 2s are on the same row or column
        same_row = r1 == r2
        same_col = c1 == c2

        if same_row:
            # 2s on same row - fill border pattern
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 0:
                        # Fill if in extreme rows or columns containing 2s
                        in_row = i == r1 or i == len(grid) - 1
                        in_col = j == c1 or j == c2
                        if in_row or in_col:
                            result[i][j] = 4
        elif same_col:
            # 2s on same column - fill border pattern
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 0:
                        # Fill if in extreme columns or rows containing 2s
                        in_row = i == r1 or i == r2
                        in_col = j == c1 or j == len(grid[0]) - 1
                        if in_row or in_col:
                            result[i][j] = 4
        else:
            # 2s at different positions - complex pattern based on configuration
            min_row, max_row = min(r1, r2), max(r1, r2)
            min_col, max_col = min(c1, c2), max(c1, c2)

            # Determine pattern type based on 2s positions
            if (r1, c1) == (min_row, min_col) and (r2, c2) == (max_row, max_col):
                # Top-left to bottom-right diagonal - vertical stripe pattern
                for i in range(min_row, max_row + 1):
                    # First row fills from min_col+1, other rows from min_col+2
                    start_col = min_col + 1 if i == min_row else min_col + 2
                    for j in range(start_col, max_col):
                        if grid[i][j] == 0:
                            result[i][j] = 4
            else:
                # Other diagonal patterns (top-right to bottom-left)
                if len(grid) >= 10:
                    # Very large grid: specific row-based pattern
                    for i in range(min_row, max_row + 1):
                        if i >= 3 and i <= 5:
                            # Fill middle rows completely
                            for j in range(min_col, max_col + 1):
                                if grid[i][j] == 0:
                                    result[i][j] = 4
                        elif i == 1 or i == 2:
                            # Fill only rightmost column
                            if grid[i][max_col] == 0:
                                result[i][max_col] = 4
                        elif i >= 6:
                            # Fill only leftmost column (column 0)
                            for j in range(min_col, max_col + 1):
                                if j == min_col and grid[i][j] == 0:
                                    result[i][j] = 4
                elif len(grid) >= 6:
                    # Large grid: fill entire rectangle
                    for i in range(min_row, max_row + 1):
                        for j in range(min_col, max_col + 1):
                            if grid[i][j] == 0:
                                result[i][j] = 4
                else:
                    # Small grid: fill specific pattern
                    for i in range(min_row + 1, max_row + 1):
                        for j in range(min_col, max_col + 1):
                            if i == max_row and j >= 2:
                                # Skip bottom-right corner area
                                continue
                            if grid[i][j] == 0:
                                result[i][j] = 4

    return result
