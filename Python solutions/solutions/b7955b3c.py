def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with scattered 8s that act as "holes" or "noise"
    2. Output is the same grid with all 8s replaced by contextually appropriate values
    3. Each 8 should be replaced based on neighboring non-8 values
    4. This is an inpainting problem where holes are filled iteratively
    5. Horizontal neighbors (left/right) are prioritized over vertical ones (up/down)

    Procedure:
    1. Iterate through the grid repeatedly until all 8s are replaced
    2. For each cell with value 8:
       a. Collect 4-connected neighbors (left, right, up, down)
       b. Filter out any neighbors that are also 8
       c. Count occurrences of each non-8 neighbor value
       d. Replace the 8 with the most common value
       e. When tied, use the first occurrence (horizontal neighbors appear first)
    3. Repeat until no more changes occur
    """

    result = [row[:] for row in grid]
    rows, cols = len(result), len(result[0])

    # Iteratively fill holes
    changed = True
    while changed:
        changed = False
        new_result = [row[:] for row in result]

        for i in range(rows):
            for j in range(cols):
                if result[i][j] == 8:
                    # Collect all 4-connected neighbors
                    neighbors = []

                    # Horizontal neighbors (left, right)
                    for di, dj in [(0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if result[ni][nj] != 8:
                                neighbors.append(result[ni][nj])

                    # Vertical neighbors (up, down)
                    for di, dj in [(-1, 0), (1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if result[ni][nj] != 8:
                                neighbors.append(result[ni][nj])

                    if neighbors:
                        from collections import Counter
                        counts = Counter(neighbors)
                        max_count = max(counts.values())

                        # Pick first occurrence of most common value (horizontal priority)
                        for val in neighbors:
                            if counts[val] == max_count:
                                most_common_value = val
                                break

                        new_result[i][j] = most_common_value
                        changed = True

        result = new_result

    return result
