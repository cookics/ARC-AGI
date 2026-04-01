def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid has a placeholder value that forms rectangular blocks
    2. Placeholder cells differ from their symmetric reflections
    3. Use symmetry reflections to fill placeholder cells

    Procedure:
    1. Detect placeholder value (forms blocks and differs from reflections)
    2. For each placeholder cell, check symmetry reflections
    3. Fill with consensus value from reflections
    4. Iterate until convergence
    """
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Detect placeholder: find value that forms largest rectangular block
    def find_largest_rect(val):
        max_area = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == val:
                    # Find max rectangle starting at (r, c)
                    h = 0
                    while r + h < rows and grid[r + h][c] == val:
                        h += 1
                    for hh in range(1, h + 1):
                        w = 0
                        while c + w < cols and all(grid[r + i][c + w] == val for i in range(hh) if r + i < rows):
                            w += 1
                        max_area = max(max_area, hh * w)
        return max_area

    # Get all unique values
    all_values = set()
    for row in grid:
        all_values.update(row)

    # Find value with largest rectangular block
    placeholder = max(all_values, key=find_largest_rect)

    result = [row[:] for row in grid]

    # Iteratively fill placeholders using symmetry
    for iteration in range(30):
        changed = False

        for r in range(rows):
            for c in range(cols):
                if result[r][c] != placeholder:
                    continue

                candidates = []

                # Point reflection (180° rotation)
                pr, pc = rows - 1 - r, cols - 1 - c
                if 0 <= pr < rows and 0 <= pc < cols and result[pr][pc] != placeholder:
                    candidates.append(result[pr][pc])

                # Horizontal reflection
                pr, pc = r, cols - 1 - c
                if 0 <= pr < rows and 0 <= pc < cols and result[pr][pc] != placeholder:
                    candidates.append(result[pr][pc])

                # Vertical reflection
                pr, pc = rows - 1 - r, c
                if 0 <= pr < rows and 0 <= pc < cols and result[pr][pc] != placeholder:
                    candidates.append(result[pr][pc])

                # Use most common value from reflections
                if candidates:
                    result[r][c] = Counter(candidates).most_common(1)[0][0]
                    changed = True

        if not changed:
            break

    # Fallback: use neighbors
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == placeholder:
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] != placeholder:
                        neighbors.append(result[nr][nc])

                if neighbors:
                    result[r][c] = Counter(neighbors).most_common(1)[0][0]

    return result
