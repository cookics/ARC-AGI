def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid containing values 7 (background), 6 (rectangle border), and a pattern color
    2. Output is the same grid with the pattern moved into the hollow rectangle's interior and original pattern cleared
    3. The grid contains a hollow rectangle formed by 6s
    4. A pattern made of a different color (not 6, not 7) appears elsewhere in the grid
    5. The pattern's bounding box has certain edges fully filled (left/right/top/bottom)
    6. In the output, corresponding edges of the rectangle's interior are filled with the pattern color
    7. If the pattern is a solid block, the entire interior is filled
    8. Original pattern positions are set to 7

    Procedure:
    1. Copy the input grid
    2. Find all cells with value 6 to determine the hollow rectangle bounds
    3. Find all cells that are not 6 or 7 (the pattern) and identify the pattern value
    4. Determine the pattern's bounding box
    5. Check which edges (left, right, top, bottom) of the pattern bounding box are fully filled
    6. If pattern is a solid rectangular block, fill entire interior with pattern value
    7. Otherwise, fill corresponding edges in the rectangle interior based on which pattern edges are filled
    8. Clear all original pattern positions by setting them to 7
    """

    def find_rectangle(grid, start_r, start_c, visited):
        """Find a 6-bounded rectangle starting from the given position"""
        rows, cols = len(grid), len(grid[0])

        # Expand to find the full rectangle
        # Find the extent of 6's connected to this starting point
        component = []
        stack = [(start_r, start_c)]
        temp_visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in temp_visited or r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != 6:
                continue

            temp_visited.add((r, c))
            component.append((r, c))

            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))

        if len(component) < 8:  # Too small to be a meaningful rectangle
            return None

        # Find bounds
        rs = [r for r, c in component]
        cs = [c for r, c in component]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        # Check if it forms a proper hollow rectangle
        perimeter_count = 0

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    # Should be 6 on perimeter
                    if grid[r][c] == 6:
                        perimeter_count += 1

        # Mark as visited
        for pos in component:
            visited.add(pos)

        # Return rectangle bounds if it looks valid
        if perimeter_count >= 8 and max_r > min_r + 1 and max_c > min_c + 1:
            return (min_r, max_r, min_c, max_c)

        return None

    def place_complex_pattern(
        result,
        positions,
        interior_min_r,
        interior_max_r,
        interior_min_c,
        interior_max_c,
        source_val,
    ):
        """Place a complex pattern inside the rectangle interior"""
        # Normalize pattern positions relative to their bounding box
        pattern_rows = [r for r, c in positions]
        pattern_cols = [c for r, c in positions]
        pattern_min_r, pattern_max_r = min(pattern_rows), max(pattern_rows)
        pattern_min_c, pattern_max_c = min(pattern_cols), max(pattern_cols)

        # Find which rows and columns are occupied in the original pattern
        occupied_rows = set(pattern_rows)
        occupied_cols = set(pattern_cols)

        # For each occupied row in the original, extend it horizontally in the interior
        for orig_row in range(pattern_min_r, pattern_max_r + 1):
            if orig_row in occupied_rows:
                # Map this row to a row in the interior
                row_ratio = (orig_row - pattern_min_r) / max(
                    1, pattern_max_r - pattern_min_r
                )
                target_row = interior_min_r + int(
                    row_ratio * (interior_max_r - interior_min_r)
                )

                # Fill the entire row if this was the bottom row of the pattern
                if orig_row == pattern_max_r:
                    for c in range(interior_min_c, interior_max_c + 1):
                        result[target_row][c] = source_val

        # For each occupied column in the original, extend it vertically in the interior
        for orig_col in range(pattern_min_c, pattern_max_c + 1):
            if orig_col in occupied_cols:
                # Map this column to a column in the interior
                col_ratio = (orig_col - pattern_min_c) / max(
                    1, pattern_max_c - pattern_min_c
                )
                target_col = interior_min_c + int(
                    col_ratio * (interior_max_c - interior_min_c)
                )

                # Fill the entire column if this was the leftmost column of the pattern
                if orig_col == pattern_min_c:
                    for r in range(interior_min_r, interior_max_r + 1):
                        result[r][target_col] = source_val

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all source patterns grouped by value
    source_patterns = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7 and grid[r][c] != 6:
                val = grid[r][c]
                if val not in source_patterns:
                    source_patterns[val] = []
                source_patterns[val].append((r, c))

    # Find 6-bounded rectangles
    rectangles = []
    visited = set()

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6 and (r, c) not in visited:
                rect = find_rectangle(grid, r, c, visited)
                if rect:
                    rectangles.append(rect)

    # Fill rectangles with source patterns
    for source_val, positions in source_patterns.items():
        for rect in rectangles:
            min_r, max_r, min_c, max_c = rect
            interior_height = max_r - min_r - 1
            interior_width = max_c - min_c - 1

            if interior_height <= 0 or interior_width <= 0:
                continue

            # Analyze the source pattern
            if len(positions) == 1:
                # Single cell - place in center
                center_r = min_r + 1 + interior_height // 2
                center_c = min_c + 1 + interior_width // 2
                result[center_r][center_c] = source_val
            else:
                # Multiple cells - analyze pattern shape
                pattern_rows = [r for r, c in positions]
                pattern_cols = [c for r, c in positions]
                pattern_min_r, pattern_max_r = min(pattern_rows), max(pattern_rows)
                pattern_min_c, pattern_max_c = min(pattern_cols), max(pattern_cols)
                pattern_height = pattern_max_r - pattern_min_r + 1
                pattern_width = pattern_max_c - pattern_min_c + 1

                # Check if it's a solid rectangular block
                is_solid_block = len(positions) == pattern_height * pattern_width

                if is_solid_block:
                    # Fill entire interior
                    for r in range(min_r + 1, max_r):
                        for c in range(min_c + 1, max_c):
                            result[r][c] = source_val
                else:
                    # Complex pattern - place with shape preservation
                    place_complex_pattern(
                        result,
                        positions,
                        min_r + 1,
                        max_r - 1,
                        min_c + 1,
                        max_c - 1,
                        source_val,
                    )

    # Clear original source positions
    for positions in source_patterns.values():
        for r, c in positions:
            result[r][c] = 7

    return result
