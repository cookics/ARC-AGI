def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid has two colors + background
    2. Find cells that are "wrong" for their neighborhood
    3. Apply cross pattern: center=8, 4-neighbors=fill_color

    Procedure:
    1. Find two main colors
    2. For each cell, check if it's minority in 4-neighborhood
    3. Apply cross transformations
    """

    rows = len(grid)
    cols = len(grid[0])

    from collections import Counter

    # Find the two main non-zero values
    colors = [grid[r][c] for r in range(rows) for c in range(cols) if grid[r][c] != 0]

    if len(set(colors)) < 2:
        return [row[:] for row in grid]

    color_counts = Counter(colors)
    two_colors = [c for c, _ in color_counts.most_common(2)]
    colorA, colorB = sorted(two_colors)

    result = [row[:] for row in grid]

    # Strategy: Find cells that don't match their 4-neighborhood majority
    crosses_to_apply = []

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]

            # Count 4-neighborhood only
            neighbors_4 = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors_4.append(grid[nr][nc])

            if not neighbors_4:
                continue

            # Count values in 4-neighborhood
            count_val = neighbors_4.count(val)
            countA = neighbors_4.count(colorA)
            countB = neighbors_4.count(colorB)
            count0 = neighbors_4.count(0)

            # Check if cell is anomalous (minority)
            if val == 0:
                # 0 with mostly colored neighbors
                if countA + countB >= 3 and countA != countB:
                    fill_color = colorB if countA > countB else colorA
                    crosses_to_apply.append((r, c, fill_color))
            else:
                # Colored cell with mostly different value
                if val == colorA and countB >= 3:
                    crosses_to_apply.append((r, c, colorB))
                elif val == colorB and countA >= 3:
                    crosses_to_apply.append((r, c, colorA))

    # Apply crosses
    hole_centers = set((r, c) for r, c, _ in crosses_to_apply)

    for r, c, fill_color in crosses_to_apply:
        result[r][c] = 8

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in hole_centers:
                    result[nr][nc] = fill_color

    return result
