def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a border frame of a specific color
    2. Output has separators (3s) extended to full lines (horizontal or vertical)
    3. For vertical separators: rows with 2s get filled with border color, rows without get 0s
    4. For horizontal separators: columns with 2s get filled with border color, columns without get 0s
    5. Values 2 and 3 are always preserved in the output

    Procedure:
    1. Identify border color from corner cell
    2. Detect separator orientation by counting 3s in rows vs columns
    3. For vertical separators: extend 3s vertically, process rows based on 2 content
    4. For horizontal separators: extend 3s horizontally, process columns based on 2 content
    5. Fill cells appropriately while preserving 2s and 3s
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Identify border color
    border_color = grid[0][0]

    # Find the 3-structure
    three_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                three_positions.append((r, c))

    if not three_positions:
        return result

    # Determine structure type
    three_rows = set(r for r, c in three_positions)
    three_cols = set(c for r, c in three_positions)

    # Check if any row has 3s spanning most of the width (horizontal structure)
    is_horizontal = False
    for r in three_rows:
        # Check if most of the row (excluding borders) is 3s
        threes_in_row = sum(1 for c in range(1, cols - 1) if grid[r][c] == 3)
        if threes_in_row >= cols - 3:  # Most of row is 3s (allowing for borders)
            is_horizontal = True
            break

    if not is_horizontal and len(three_cols) >= 2:  # Vertical structure
        left_col = min(three_cols)
        right_col = max(three_cols)

        for r in range(rows):
            # Check if this is a border row (all border color)
            is_border_row = all(grid[r][c] == border_color for c in range(cols))

            if is_border_row:
                # Border rows: outside→0, structure→3, inside→0
                for c in range(cols):
                    if c == left_col or c == right_col:
                        result[r][c] = 3
                    else:
                        result[r][c] = 0
            else:
                # Check what's inside the structure
                has_non_zero_inside = False
                for c in range(left_col + 1, right_col):
                    if grid[r][c] != 0:
                        has_non_zero_inside = True
                        break

                # Transform based on inside content
                for c in range(cols):
                    if c == left_col or c == right_col:
                        # Keep 3-structure
                        result[r][c] = 3
                    elif c < left_col or c > right_col:
                        # Outside structure
                        if has_non_zero_inside:
                            # 2s inside: 0s become border color
                            if grid[r][c] == 0:
                                result[r][c] = border_color
                        else:
                            # 0s inside: border color becomes 0
                            if grid[r][c] == border_color:
                                result[r][c] = 0
                    # Inside structure remains unchanged

    elif is_horizontal:  # Horizontal structure
        top_row = min(three_rows)
        bottom_row = max(three_rows)

        # Get content pattern from middle rows (remove border elements)
        content_pattern = None
        if top_row + 1 < bottom_row:
            sample_row = grid[top_row + 1]
            content_pattern = sample_row[1:-1]  # Remove first and last border elements

        for r in range(rows):
            if r == top_row or r == bottom_row:
                # 3-boundary rows become all 3s
                result[r] = [3] * cols
            elif top_row < r < bottom_row:
                # Content rows: start with 0, use pattern, extend with final element
                result[r] = [0]  # First element is 0
                for c in range(len(content_pattern)):
                    result[r].append(content_pattern[c])
                # Add final element (extend pattern)
                if content_pattern and content_pattern[0] == 2:
                    result[r].append(2)
                else:
                    result[r].append(0)
            else:
                # Other rows: transform content pattern for empty areas
                result[r] = [0]  # Start with 0
                # Transform content pattern: 2→border_color, 0→0
                for c in range(len(content_pattern)):
                    if content_pattern[c] == 2:
                        result[r].append(border_color)
                    else:
                        result[r].append(0)
                # Add final element
                result[r].append(border_color)

    return result
