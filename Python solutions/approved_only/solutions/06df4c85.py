def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with separator lines dividing it into cells.
    2. Output shows colors propagating within rows and columns of cells.
    3. When a color appears multiple times in the same row or column of cells, it fills intermediate cells.
    4. Colors only propagate between their original boundary positions.

    Procedure:
    1. Identify the separator color by finding the most frequent non-zero value.
    2. Find cell boundaries by detecting complete separator lines.
    3. Extract colors from each cell and group them by their positions.
    4. Apply horizontal propagation for colors appearing multiple times in same row.
    5. Apply vertical propagation for colors appearing multiple times in same column.
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find separator color (appears most frequently and forms grid lines)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1

    # Separator is the most frequent non-zero color
    separator = max(
        (color for color in color_counts if color != 0), key=lambda x: color_counts[x]
    )

    # Find cell boundaries by detecting separator lines
    cell_row_bounds = [0]  # Start with grid boundary
    cell_col_bounds = [0]  # Start with grid boundary

    # Find horizontal separator lines (rows that are all separator color)
    for r in range(rows):
        if all(grid[r][c] == separator for c in range(cols)):
            cell_row_bounds.append(r)

    # Find vertical separator lines (columns that are all separator color)
    for c in range(cols):
        if all(grid[r][c] == separator for r in range(rows)):
            cell_col_bounds.append(c)

    # Add grid boundaries at the end
    cell_row_bounds.append(rows)
    cell_col_bounds.append(cols)

    # Parse cells and extract colors
    cell_contents = {}  # (cell_row, cell_col) -> set of colors

    for i in range(len(cell_row_bounds) - 1):
        for j in range(len(cell_col_bounds) - 1):
            # Handle grid boundaries vs separator boundaries
            r_start = cell_row_bounds[i] + (1 if cell_row_bounds[i] > 0 else 0)
            r_end = cell_row_bounds[i + 1]
            c_start = cell_col_bounds[j] + (1 if cell_col_bounds[j] > 0 else 0)
            c_end = cell_col_bounds[j + 1]

            # Find non-zero, non-separator colors in this cell
            colors_in_cell = set()
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if 0 <= r < rows and 0 <= c < cols:
                        color = grid[r][c]
                        if color != 0 and color != separator:
                            colors_in_cell.add(color)

            if colors_in_cell:
                cell_contents[(i, j)] = colors_in_cell

    # Group cells by color
    color_positions = {}  # color -> list of (cell_row, cell_col)
    for (cell_r, cell_c), colors in cell_contents.items():
        for color in colors:
            if color not in color_positions:
                color_positions[color] = []
            color_positions[color].append((cell_r, cell_c))

    # Apply propagation for each color
    for color, positions in color_positions.items():
        # Group by row and column
        rows_with_color = {}  # row -> list of columns
        cols_with_color = {}  # col -> list of rows

        for cell_r, cell_c in positions:
            if cell_r not in rows_with_color:
                rows_with_color[cell_r] = []
            rows_with_color[cell_r].append(cell_c)

            if cell_c not in cols_with_color:
                cols_with_color[cell_c] = []
            cols_with_color[cell_c].append(cell_r)

        # Horizontal propagation (within rows that have multiple instances)
        for row, columns in rows_with_color.items():
            if len(columns) > 1:
                min_col, max_col = min(columns), max(columns)
                for col in range(min_col, max_col + 1):
                    # Fill this cell with the color
                    r_start = cell_row_bounds[row] + (
                        1 if cell_row_bounds[row] > 0 else 0
                    )
                    r_end = cell_row_bounds[row + 1]
                    c_start = cell_col_bounds[col] + (
                        1 if cell_col_bounds[col] > 0 else 0
                    )
                    c_end = cell_col_bounds[col + 1]

                    for r in range(r_start, r_end):
                        for c in range(c_start, c_end):
                            if 0 <= r < rows and 0 <= c < cols and result[r][c] == 0:
                                result[r][c] = color

        # Vertical propagation (within columns that have multiple instances)
        for col, rows_list in cols_with_color.items():
            if len(rows_list) > 1:
                min_row, max_row = min(rows_list), max(rows_list)
                for row in range(min_row, max_row + 1):
                    # Fill this cell with the color
                    r_start = cell_row_bounds[row] + (
                        1 if cell_row_bounds[row] > 0 else 0
                    )
                    r_end = cell_row_bounds[row + 1]
                    c_start = cell_col_bounds[col] + (
                        1 if cell_col_bounds[col] > 0 else 0
                    )
                    c_end = cell_col_bounds[col + 1]

                    for r in range(r_start, r_end):
                        for c in range(c_start, c_end):
                            if 0 <= r < rows and 0 <= c < cols and result[r][c] == 0:
                                result[r][c] = color

    return result
