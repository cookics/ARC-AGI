def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has patterns on one edge, output moves them to the opposite edge
    2. Background is 7, separators/lines are 0, markers are 9
    3. Two types: vertical transformation (with vertical separators) or horizontal
    4. For horizontal: rows without 0s stay unchanged, rows with 0s get transformed
    5. For vertical: sections with patterns move from one end to the other

    Procedure:
    1. Detect vertical separator columns (many 0s)
    2. If vertical separators exist → vertical transformation
    3. Otherwise → horizontal transformation (row-by-row)
    4. For each row with 0s: extract pattern from source edge, move to opposite edge
    """
    rows = len(grid)
    cols = len(grid[0])

    # Detect vertical separator columns (>70% zeros)
    separator_cols = set()
    for c in range(cols):
        zero_count = sum(1 for r in range(rows) if grid[r][c] == 0)
        if zero_count >= rows * 0.7:
            separator_cols.add(c)

    if len(separator_cols) >= 2:
        # ===== VERTICAL TRANSFORMATION =====
        # Find rows with non-separator patterns
        pattern_rows = []
        for r in range(rows):
            for c in range(cols):
                if c not in separator_cols and grid[r][c] != 7:
                    pattern_rows.append(r)
                    break

        if not pattern_rows:
            return [[7] * cols for _ in range(rows)]

        result = [[7] * cols for _ in range(rows)]

        # Check if patterns are at top or bottom
        if min(pattern_rows) < rows // 2:
            # Patterns at top, move to bottom
            target_start = rows - len(pattern_rows)
            for i, src_r in enumerate(pattern_rows):
                dest_r = target_start + i
                for c in range(cols):
                    if c in separator_cols and grid[src_r][c] == 0:
                        result[dest_r][c] = 0
                    elif c not in separator_cols:
                        result[dest_r][c] = grid[src_r][c]
            # Keep 9s in their column positions
            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == 9:
                        result[r][c] = 9
        else:
            # Patterns at bottom, move to top
            for i, src_r in enumerate(pattern_rows):
                for c in range(cols):
                    result[i][c] = grid[src_r][c]

            # Handle duplicate 9s within the moved section
            seen_nine = False
            for i in range(len(pattern_rows)):
                for c in range(cols):
                    if result[i][c] == 9:
                        if seen_nine:
                            # This is a duplicate 9, remove from moved section
                            result[i][c] = 7
                            # Place it in its original row position
                            result[pattern_rows[i]][c] = 9
                        else:
                            seen_nine = True

        return result

    else:
        # ===== HORIZONTAL TRANSFORMATION =====
        result = [[7] * cols for _ in range(rows)]

        # Detect direction: count 0s on left vs right
        left_zeros = sum(1 for r in range(rows) for c in range(min(3, cols)) if grid[r][c] == 0)
        right_zeros = sum(1 for r in range(rows) for c in range(max(0, cols-3), cols) if grid[r][c] == 0)

        if left_zeros > right_zeros:
            # LEFT TO RIGHT transformation
            for r in range(rows):
                if 0 not in grid[r]:
                    # No 0s → keep row as is
                    result[r] = grid[r][:]
                else:
                    # Has 0s → extract and move to right
                    zero_count = grid[r].count(0)

                    if zero_count >= cols // 2:
                        # Long line of 0s: extract prefix before the line
                        prefix = []
                        for c in range(cols):
                            if grid[r][c] == 0:
                                break
                            prefix.append(grid[r][c])

                        # If prefix has non-7 values, move them + one 0
                        if prefix and any(v != 7 for v in prefix):
                            start_pos = cols - len(prefix) - 1
                            for i, val in enumerate(prefix):
                                result[r][start_pos + i] = val
                            result[r][cols - 1] = 0
                        else:
                            # Just one 0 at end
                            result[r][cols - 1] = 0
                    else:
                        # Short pattern: extract all non-7 from left
                        pattern = []
                        for c in range(cols):
                            if grid[r][c] != 7:
                                pattern.append(grid[r][c])
                            elif pattern:
                                break

                        if pattern:
                            start_pos = cols - len(pattern)
                            for i, val in enumerate(pattern):
                                result[r][start_pos + i] = val

        else:
            # RIGHT TO LEFT transformation
            for r in range(rows):
                if 0 not in grid[r]:
                    # No 0s → keep row as is
                    result[r] = grid[r][:]
                else:
                    # Has 0s → extract and move to left
                    zero_count = grid[r].count(0)

                    if zero_count >= cols // 2:
                        # Long line of 0s: extract suffix after the line
                        suffix = []
                        for c in range(cols - 1, -1, -1):
                            if grid[r][c] != 0:
                                suffix.insert(0, grid[r][c])
                            else:
                                if suffix:
                                    break

                        # Move one 0 + suffix to left
                        if suffix:
                            result[r][0] = 0
                            for i, val in enumerate(suffix):
                                result[r][i + 1] = val
                        else:
                            result[r][0] = 0
                    else:
                        # Short pattern: extract all non-7 from right
                        pattern = []
                        for c in range(cols - 1, -1, -1):
                            if grid[r][c] != 7:
                                pattern.insert(0, grid[r][c])
                            elif pattern:
                                break

                        if pattern:
                            for i, val in enumerate(pattern):
                                result[r][i] = val

        return result
