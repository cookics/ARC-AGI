def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a background color, a pattern color, and a marker (4s in last row)
    2. The marker defines a column range
    3. Transformation based on row_index % 6:
       - row % 6 == 0 and row == 0: place 3s at marker position
       - row % 6 == 0 and row > 0: replace pattern cells with 3s (spanning their extent)
       - row % 6 in {2, 4}: place 4s at marker position
       - other rows: no special markers

    Procedure:
    1. Find marker position (last row with 4s)
    2. Identify background and pattern colors
    3. For each row, apply transformation based on row_index % 6
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find marker (4s in last row or any row)
    marker_cols = []
    for r in range(rows - 1, -1, -1):
        for c in range(cols):
            if grid[r][c] == 4:
                marker_cols.append(c)
        if marker_cols:
            break

    if not marker_cols:
        return result

    marker_min, marker_max = min(marker_cols), max(marker_cols)

    # Determine background and pattern colors
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 4:
                color_counts[grid[r][c]] = color_counts.get(grid[r][c], 0) + 1

    sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
    background = sorted_colors[0][0]
    pattern = sorted_colors[1][0] if len(sorted_colors) > 1 else background

    # Process each row based on row_index % 6
    for r in range(rows):
        mod = r % 6

        if mod == 0:
            if r == 0:
                # Place 3s at marker position
                for c in range(marker_min, marker_max + 1):
                    result[r][c] = 3
            else:
                # Replace all pattern cells in this row with 3s
                for c in range(cols):
                    if grid[r][c] == pattern:
                        result[r][c] = 3
                # Also place 3s at marker position
                for c in range(marker_min, marker_max + 1):
                    result[r][c] = 3

        elif mod in {2, 4}:
            # Place 4s at marker position (overwrite anything except existing 4s)
            for c in range(marker_min, marker_max + 1):
                if result[r][c] != 4:
                    result[r][c] = 4

        elif mod in {1, 3, 5}:
            # Remove pattern cells within marker range (replace with background)
            for c in range(marker_min, marker_max + 1):
                if result[r][c] == pattern:
                    result[r][c] = background

    return result
