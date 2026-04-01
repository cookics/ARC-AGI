def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with scattered non-zero colored cells
    2. Output is a grid with pyramid shapes at the bottom
    3. Each color falls down to form a shape centered on its original column
    4. If a color has k² cells (perfect square), it forms a k-row pyramid with widths 1, 3, 5, ..., 2k-1
    5. Otherwise, it forms a single horizontal row with that many cells
    6. Colors are stacked by their first appearance order (top to bottom in input)

    Procedure:
    1. Find all colored cells, count each color, and track first appearance position
    2. Sort colors by first appearance row
    3. Create output grid filled with zeros
    4. For each color in order, calculate shape size (pyramid rows or single row)
    5. Place each color's shape at the bottom of the grid, stacking upward
    6. Center each shape on the column where that color first appeared
    """
    rows, cols = len(grid), len(grid[0])

    # Find all colored cells and their properties
    color_info = {}  # color -> {count, first_row, first_col}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_info:
                    color_info[color] = {"count": 0, "first_row": r, "first_col": c}
                color_info[color]["count"] += 1

    # Sort colors by first appearance row
    sorted_colors = sorted(color_info.keys(), key=lambda c: color_info[c]["first_row"])

    # Create output grid
    result = [[0] * cols for _ in range(rows)]

    # Calculate total height needed
    total_height = 0
    for color in sorted_colors:
        count = color_info[color]["count"]
        k = int(count ** 0.5)
        if k * k == count:
            total_height += k
        else:
            total_height += 1

    # Place shapes from top to bottom of the stack (but at the bottom of grid)
    current_row = rows - total_height

    for color in sorted_colors:
        count = color_info[color]["count"]
        center_col = color_info[color]["first_col"]

        # Check if count is perfect square
        k = int(count ** 0.5)
        if k * k == count:
            # Create triangle with k rows (upside down pyramid)
            for layer in range(k):
                row = current_row + layer
                width = 2 * (layer + 1) - 1  # 1, 3, 5, 7, ...
                start_col = center_col - width // 2

                for offset in range(width):
                    col = start_col + offset
                    if 0 <= col < cols:
                        result[row][col] = color

            current_row += k
        else:
            # Create single row
            row = current_row
            start_col = center_col - count // 2

            for offset in range(count):
                col = start_col + offset
                if 0 <= col < cols:
                    result[row][col] = color

            current_row += 1

    return result
