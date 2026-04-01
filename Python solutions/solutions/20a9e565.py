def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. 5s form L-shapes marking output dimensions via bounding box
    2. Extract multi-color rectangular pattern from specific column range
    3. Hollow rows (start non-zero, rest zeros) extend with zeros
    4. Solid rows tile by repeating the pattern

    Procedure:
    1. Find 5s bounding box (H x W) for output dimensions
    2. Try all possible column starts and pattern widths that divide W
    3. Extract pattern as-is (all colors), tile with hollow row logic
    4. Select result with best coverage (most non-zero cells)
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Find 5s and output dimensions
    fives = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 5]
    if not fives:
        return [[]]

    min_r, max_r = min(r for r, c in fives), max(r for r, c in fives)
    min_c, max_c = min(c for r, c in fives), max(c for r, c in fives)
    output_height = max_r - min_r + 1
    output_width = max_c - min_c + 1

    # Find all pattern cells to determine search range
    pattern_cells = [(r, c) for r in range(rows) for c in range(cols)
                     if grid[r][c] not in (0, 5)]
    if not pattern_cells:
        return [[0] * output_width for _ in range(output_height)]

    pattern_cols = sorted(set(c for r, c in pattern_cells))
    min_pattern_col = min(pattern_cols)
    max_pattern_col = max(pattern_cols)

    best_result = None
    best_score = -1

    # Try different starting columns and widths
    for start_col in range(min_pattern_col, max_pattern_col + 1):
        for width in range(1, output_width + 1):
            if output_width % width != 0:
                continue

            # Extract pattern from 5s row range (keep all colors as-is)
            pattern_unit = []
            for r in range(min_r, max_r + 1):
                row_data = []
                for c in range(start_col, start_col + width):
                    if c < cols:
                        row_data.append(grid[r][c])
                    else:
                        row_data.append(0)
                pattern_unit.append(row_data)

            # Remove leading/trailing all-zero rows
            while pattern_unit and all(v == 0 for v in pattern_unit[0]):
                pattern_unit.pop(0)
            while pattern_unit and all(v == 0 for v in pattern_unit[-1]):
                pattern_unit.pop()

            if not pattern_unit:
                continue

            # Tile the pattern
            result = []
            for out_r in range(output_height):
                pattern_r = out_r % len(pattern_unit)
                row_template = pattern_unit[pattern_r]

                # Check if row is "hollow": starts with non-zero, has trailing zeros
                # A hollow row extends with zeros rather than tiling
                nonzero_count = sum(1 for v in row_template if v != 0)
                first_nonzero = None
                last_nonzero_idx = -1
                for i, v in enumerate(row_template):
                    if v != 0:
                        if first_nonzero is None:
                            first_nonzero = i
                        last_nonzero_idx = i

                # Hollow if: has non-zero values, but they don't extend to end
                is_hollow = (first_nonzero is not None and
                            last_nonzero_idx < len(row_template) - 1 and
                            nonzero_count < len(row_template))

                out_row = []
                for out_c in range(output_width):
                    if out_c < width:
                        out_row.append(row_template[out_c])
                    else:
                        if is_hollow:
                            # Hollow rows extend with zeros
                            out_row.append(0)
                        else:
                            # Solid rows tile by repeating
                            pattern_c = out_c % width
                            out_row.append(row_template[pattern_c])

                result.append(out_row)

            # Score this result by coverage
            nonzero_count = sum(1 for row in result for v in row if v != 0)
            total_cells = output_height * output_width

            if nonzero_count == 0:
                continue

            score = nonzero_count / total_cells

            # Prefer exact width match
            if width == output_width:
                score += 10
            else:
                score += width / output_width

            if score > best_score:
                best_score = score
                best_result = result

    if best_result:
        return best_result

    return [[0] * output_width for _ in range(output_height)]
