def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with some cells containing colored values (non-zero) and others empty (zero).
    2. Output is symmetric edge-to-edge (row 0 ↔ row n-1, column 0 ↔ column m-1).
    3. Output is also symmetric across separator lines (all-zero rows/columns).
    4. When symmetric positions differ (one 0, one non-zero), use the non-zero value.

    Procedure:
    1. Apply edge-to-edge horizontal and vertical symmetry
    2. Identify separator rows and columns (all zeros in input)
    3. Apply symmetry across each separator line
    4. Iterate until no changes (for multi-step propagation)
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Identify separator lines and center
    sep_rows = [i for i in range(rows) if all(grid[i][j] == 0 for j in range(cols))]
    sep_cols = [j for j in range(cols) if all(grid[i][j] == 0 for i in range(rows))]
    center_row = rows // 2
    center_col = cols // 2

    # Apply symmetry iteratively
    changed = True
    iterations = 0
    max_iterations = 20

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Horizontal symmetry (edge-to-edge)
        for i in range(rows):
            for j in range(cols):
                mirror_j = cols - 1 - j
                if result[i][j] == 0 and result[i][mirror_j] != 0:
                    result[i][j] = result[i][mirror_j]
                    changed = True
                elif result[i][mirror_j] == 0 and result[i][j] != 0:
                    result[i][mirror_j] = result[i][j]
                    changed = True

        # Vertical symmetry (edge-to-edge)
        for i in range(rows):
            mirror_i = rows - 1 - i
            for j in range(cols):
                if result[i][j] == 0 and result[mirror_i][j] != 0:
                    result[i][j] = result[mirror_i][j]
                    changed = True
                elif result[mirror_i][j] == 0 and result[i][j] != 0:
                    result[mirror_i][j] = result[i][j]
                    changed = True

        # Separator column symmetry (skip center row/col and edges)
        for sep_col in sep_cols:
            # Calculate adjacent region widths
            left_bound = max([c for c in sep_cols if c < sep_col] + [-1]) + 1
            right_bound = min([c for c in sep_cols if c > sep_col] + [cols]) - 1
            max_offset = min(sep_col - left_bound, right_bound - sep_col)

            for i in range(rows):
                if i in sep_rows:  # Skip separator rows
                    continue

                for offset in range(1, max_offset + 1):
                    left_j = sep_col - offset
                    right_j = sep_col + offset

                    # Skip if on center column, separator columns, or grid column edges
                    if (
                        left_j in sep_cols
                        or right_j in sep_cols
                        or left_j == center_col
                        or right_j == center_col
                        or left_j == 0
                        or right_j == cols - 1
                    ):
                        continue

                    # Skip center column at edge rows (symmetry axis intersection)
                    if (left_j == center_col or right_j == center_col) and (
                        i == 0 or i == rows - 1
                    ):
                        continue

                    if result[i][left_j] == 0 and result[i][right_j] != 0:
                        result[i][left_j] = result[i][right_j]
                        changed = True
                    elif result[i][right_j] == 0 and result[i][left_j] != 0:
                        result[i][right_j] = result[i][left_j]
                        changed = True

        # Separator row symmetry (skip center row/col and edges)
        for sep_row in sep_rows:
            # Calculate adjacent region heights
            top_bound = max([r for r in sep_rows if r < sep_row] + [-1]) + 1
            bottom_bound = min([r for r in sep_rows if r > sep_row] + [rows]) - 1
            max_offset = min(sep_row - top_bound, bottom_bound - sep_row)

            for j in range(cols):
                if (
                    j in sep_cols or j == center_col
                ):  # Skip separator and center columns
                    continue

                for offset in range(1, max_offset + 1):
                    top_i = sep_row - offset
                    bottom_i = sep_row + offset

                    # Skip if on separator rows or grid row edges
                    if (
                        top_i in sep_rows
                        or bottom_i in sep_rows
                        or top_i == 0
                        or bottom_i == rows - 1
                    ):
                        continue

                    # Skip center row at edge columns (symmetry axis intersection)
                    if (top_i == center_row or bottom_i == center_row) and (
                        j == 0 or j == cols - 1
                    ):
                        continue

                    if result[top_i][j] == 0 and result[bottom_i][j] != 0:
                        result[top_i][j] = result[bottom_i][j]
                        changed = True
                    elif result[bottom_i][j] == 0 and result[top_i][j] != 0:
                        result[bottom_i][j] = result[top_i][j]
                        changed = True

    return result
