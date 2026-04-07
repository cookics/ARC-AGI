def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a dominant value that forms a large region
    2. This region has gaps that should be filled based on proximity
    3. The filling happens row by row, connecting gaps between dominant value occurrences
    4. Rows with higher concentration of dominant value get more extensive filling

    Procedure:
    1. Find the most common non-zero value (dominant value)
    2. For each row, find where dominant values appear
    3. Fill gaps between dominant values if the gap is small enough
    4. Use iterative propagation to fill cells with many dominant neighbors
    """
    from collections import Counter
    import copy

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = copy.deepcopy(grid)

    # Find the dominant non-zero value
    value_counts = Counter()
    for row in grid:
        for val in row:
            if val != 0:
                value_counts[val] += 1

    if not value_counts:
        return result

    dominant_value = value_counts.most_common(1)[0][0]

    # Iterative filling based on neighbor count
    # Fill cells that have enough neighbors with the dominant value
    for iteration in range(30):
        changed = False
        for r in range(rows):
            for c in range(cols):
                if result[r][c] == dominant_value:
                    continue

                # Count neighbors with dominant value (orthogonal only)
                ortho_count = 0
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if result[nr][nc] == dominant_value:
                            ortho_count += 1

                # Count diagonal neighbors
                diag_count = 0
                for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if result[nr][nc] == dominant_value:
                            diag_count += 1

                # Fill if: 4 orthogonal neighbors, OR 3 orthogonal + 2 diagonal
                if ortho_count == 4 or (ortho_count == 3 and diag_count >= 2):
                    result[r][c] = dominant_value
                    changed = True

        if not changed:
            break

    return result
