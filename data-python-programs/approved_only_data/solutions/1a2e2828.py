def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid of integers with 0s representing empty cells and non-zero values representing colored cells.
    2. Output is a single integer wrapped in a 2D array format [[value]].
    3. Each grid contains vertical lines (columns where the same non-zero value spans multiple rows).
    4. Each grid contains horizontal lines (rows where the same non-zero value spans multiple columns).
    5. The target value forms a complete line that spans the entire width or height of the grid.
    6. Complete lines contain only one unique non-zero value across their entire span.

    Procedure:
    1. Find all horizontal lines by checking each row for values that span the entire width.
    2. Find all vertical lines by checking each column for values that span the entire height.
    3. Look for values that appear in both horizontal and vertical complete lines (intersection).
    4. If intersection exists, return that value.
    5. Otherwise, return any value that forms a complete line (horizontal or vertical).
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find horizontal lines that span the full width
    horizontal_line_values = set()
    for r in range(rows):
        # Check if this row has a value that spans the entire width
        row_values = [grid[r][c] for c in range(cols) if grid[r][c] != 0]
        if len(set(row_values)) == 1 and len(row_values) == cols:
            horizontal_line_values.add(row_values[0])

    # Find vertical lines that span the full height
    vertical_line_values = set()
    for c in range(cols):
        # Check if this column has a value that spans the entire height
        col_values = [grid[r][c] for r in range(rows) if grid[r][c] != 0]
        if len(set(col_values)) == 1 and len(col_values) == rows:
            vertical_line_values.add(col_values[0])

    # Find the intersection of horizontal and vertical line values
    intersection = horizontal_line_values.intersection(vertical_line_values)

    if intersection:
        return [[list(intersection)[0]]]

    # If no intersection, return the value that forms any complete line
    all_line_values = horizontal_line_values.union(vertical_line_values)
    if all_line_values:
        return [[list(all_line_values)[0]]]

    # Fallback - this shouldn't happen based on the examples
    return [[0]]
