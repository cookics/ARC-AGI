def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing zeros and non-zero values representing different colors.
    2. The grid contains horizontal or vertical "lines" (complete rows/columns of the same non-zero value).
    3. There are scattered individual cells with values that may or may not match the line colors.
    4. The output shows scattered cells that match line colors moved to be adjacent to those lines.
    5. Scattered cells that don't match any line color disappear in the output.

    Procedure:
    1. Find all horizontal lines (rows where all values are the same non-zero value).
    2. Find all vertical lines (columns where all values are the same non-zero value).
    3. For each scattered cell that matches a line color, move it to be adjacent to that line.
    4. If horizontal line: move to row above/below the line based on original position.
    5. If vertical line: move to column left/right of the line based on original position.
    6. Clear all other scattered values that don't match any line.
    """
    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find horizontal lines
    horizontal_lines = {}  # color -> row_index
    for r in range(rows):
        if all(grid[r][c] == grid[r][0] for c in range(cols)) and grid[r][0] != 0:
            horizontal_lines[grid[r][0]] = r
            # Copy the line to result
            for c in range(cols):
                result[r][c] = grid[r][c]

    # Find vertical lines
    vertical_lines = {}  # color -> col_index
    for c in range(cols):
        if all(grid[r][c] == grid[0][c] for r in range(rows)) and grid[0][c] != 0:
            vertical_lines[grid[0][c]] = c
            # Copy the line to result
            for r in range(rows):
                result[r][c] = grid[r][c]

    # Process scattered cells
    for r in range(rows):
        for c in range(cols):
            cell_value = grid[r][c]
            if cell_value == 0:
                continue

            # Check if this cell is part of a line
            is_part_of_line = False
            if cell_value in horizontal_lines and r == horizontal_lines[cell_value]:
                is_part_of_line = True
            if cell_value in vertical_lines and c == vertical_lines[cell_value]:
                is_part_of_line = True

            if not is_part_of_line:
                # This is a scattered cell
                if cell_value in horizontal_lines:
                    # Move to adjacent row
                    line_row = horizontal_lines[cell_value]
                    if r < line_row:
                        # Above the line, move to row above line
                        new_row = line_row - 1
                    else:
                        # Below the line, move to row below line
                        new_row = line_row + 1
                    result[new_row][c] = cell_value

                elif cell_value in vertical_lines:
                    # Move to adjacent column
                    line_col = vertical_lines[cell_value]
                    if c < line_col:
                        # Left of the line, move to column left of line
                        new_col = line_col - 1
                    else:
                        # Right of the line, move to column right of line
                        new_col = line_col + 1
                    result[r][new_col] = cell_value

                # If cell_value doesn't match any line, it disappears (already 0 in result)

    return result
