def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid with non-zero values forming shapes.
    2. The output transforms these shapes by shifting most elements right by 1 position.
    3. The bottom row of each shape stays in its original position.
    4. Elements at the rightmost column of each shape don't shift beyond that boundary.
    5. Multiple shapes can exist in the same grid and are processed independently.

    Procedure:
    1. Group all non-zero cells by their value to identify separate shapes.
    2. For each shape, find the bottom row and rightmost column boundaries.
    3. Keep the bottom row cells in their original positions.
    4. Shift all other rows right by 1, but don't exceed the rightmost column boundary.
    5. Place the transformed values into the output grid.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Find all values and group cells by value (not by connectivity)
    value_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                value = grid[r][c]
                if value not in value_cells:
                    value_cells[value] = []
                value_cells[value].append((r, c))

    # Create output grid
    output = [[0] * cols for _ in range(rows)]

    # Process each value group as a single shape
    for value, cells in value_cells.items():
        # Group cells by row
        rows_dict = {}
        for r, c in cells:
            if r not in rows_dict:
                rows_dict[r] = []
            rows_dict[r].append(c)  # Just store column

        # Find bottom row and rightmost column of this shape
        bottom_row = max(rows_dict.keys())
        rightmost_col = max(c for r, c in cells)

        # Process each row of the shape
        for row_idx, row_cols in rows_dict.items():
            if row_idx == bottom_row:
                # Bottom row doesn't shift
                for c in row_cols:
                    output[row_idx][c] = value
            else:
                # Other rows shift right by 1, but not beyond rightmost column
                for c in row_cols:
                    new_col = c + 1
                    if new_col > rightmost_col:
                        # Don't shift if it would go beyond the rightmost column
                        output[row_idx][c] = value
                    else:
                        # Shift right by 1
                        output[row_idx][new_col] = value

    return output
