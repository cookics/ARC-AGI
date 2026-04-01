def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with background value 0 and exactly two non-zero colors.
    2. One color forms a large connected component (central structure).
    3. The other color appears as scattered individual dots throughout the grid.
    4. Output preserves the original grid and adds vertical lines using the dot color.
    5. Vertical lines are drawn from specific columns within the central structure's range.

    Procedure:
    1. Identify the two non-zero colors in the grid.
    2. Determine which color forms the largest connected component (central structure).
    3. Find the bounding box of the central structure.
    4. Identify columns within the structure's range that have scattered dots below it.
    5. For each target column, fill gaps within the structure's bounding box using the dot color.
    6. Draw vertical lines from the bottom of the central structure to the bottom of the grid.
    """

    # Copy the grid
    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find all non-background colors
    colors = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                colors.add(grid[r][c])

    if len(colors) != 2:
        return result

    # For each color, find its largest connected component size
    def get_connected_component_size(color):
        visited = set()
        max_size = 0

        def dfs(r, c):
            if (
                (r, c) in visited
                or r < 0
                or r >= rows
                or c < 0
                or c >= cols
                or grid[r][c] != color
            ):
                return 0
            visited.add((r, c))
            size = 1
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                size += dfs(r + dr, c + dc)
            return size

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in visited:
                    component_size = dfs(r, c)
                    max_size = max(max_size, component_size)

        return max_size

    color1, color2 = list(colors)
    size1 = get_connected_component_size(color1)
    size2 = get_connected_component_size(color2)

    # The central structure is the color with the larger connected component
    if size1 > size2:
        structure_color = color1
        dot_color = color2
    else:
        structure_color = color2
        dot_color = color1

    # Find the bounding box of the central structure
    structure_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == structure_color:
                structure_positions.append((r, c))

    if not structure_positions:
        return result

    structure_rows = [pos[0] for pos in structure_positions]
    structure_cols = [pos[1] for pos in structure_positions]

    min_row = min(structure_rows)
    max_row = max(structure_rows)
    min_col = min(structure_cols)
    max_col = max(structure_cols)

    # Find scattered dots below the central structure
    dots_below = []
    for r in range(max_row + 1, rows):
        for c in range(cols):
            if grid[r][c] == dot_color:
                dots_below.append((r, c))

    # Find columns with dots below that are within the central structure's column range
    target_columns = set()
    for r, c in dots_below:
        if min_col <= c <= max_col:
            target_columns.add(c)

    # For each target column:
    for col in target_columns:
        # Find the first row where structure exists in this column
        first_structure_row = None
        for r in range(min_row, max_row + 1):
            if grid[r][col] == structure_color:
                first_structure_row = r
                break

        if first_structure_row is not None:
            # Fill gaps from first structure row to bottom of structure bounding box
            for r in range(first_structure_row, max_row + 1):
                if result[r][col] == 0:
                    result[r][col] = dot_color

        # Draw vertical line from bottom of central structure to bottom of grid
        for r in range(max_row + 1, rows):
            if result[r][col] == 0:
                result[r][col] = dot_color

    return result
