def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with 5s forming various patterns (lines and hollow rectangles).
    2. Hollow rectangles have borders made entirely of 5s and interiors filled with 0s.
    3. Output fills the interior of these hollow rectangles with 4s.
    4. All other cells remain unchanged.

    Procedure:
    1. Create a deep copy of the input grid.
    2. Iterate through all possible rectangle positions and sizes.
    3. For each potential rectangle, validate it has uniform border of 5s and interior of 0s.
    4. Fill the interior with 4s.
    5. Track processed cells to avoid double-filling overlapping rectangles.
    """

    def is_valid_hollow_rectangle(grid, top, left, bottom, right, use_left_edge, use_right_edge):
        """Check if the given coordinates form a valid hollow rectangle with border of 5s
        Grid edges can act as implicit borders when use_left_edge or use_right_edge is True"""
        if top >= bottom or left >= right:
            return False

        rows, cols = len(grid), len(grid[0])
        if bottom >= rows or right >= cols:
            return False

        # Need at least 3 rows for top border + interior + bottom border
        if bottom - top < 2:
            return False

        # Need at least 2 columns when using edge as border, 3 otherwise
        min_width = 1 if (use_left_edge or use_right_edge) else 2
        if right - left < min_width:
            return False

        # Check top and bottom borders are all 5s
        for j in range(left, right + 1):
            if grid[top][j] != 5 or grid[bottom][j] != 5:
                return False

        # Check left border (if not using grid edge)
        if not use_left_edge:
            for i in range(top, bottom + 1):
                if grid[i][left] != 5:
                    return False

        # Check right border (if not using grid edge)
        if not use_right_edge:
            for i in range(top, bottom + 1):
                if grid[i][right] != 5:
                    return False

        # Determine interior bounds
        interior_left = left if use_left_edge else left + 1
        interior_right = right + 1 if use_right_edge else right

        # Check interior is all 0s
        for i in range(top + 1, bottom):
            for j in range(interior_left, interior_right):
                if grid[i][j] != 0:
                    return False

        return True

    result = [row[:] for row in grid]  # Deep copy
    rows, cols = len(grid), len(grid[0])

    # Track which cells have been processed to avoid double-filling
    processed = set()

    # Try all possible rectangles (including those using grid edges as borders)
    for top in range(rows):
        for left in range(cols):
            for bottom in range(top + 2, rows):  # Minimum height 3
                for right in range(left, cols):  # Variable width
                    # Try different border configurations
                    for use_left_edge in [False, True]:
                        for use_right_edge in [False, True]:
                            # Can only use left edge if at column 0
                            if use_left_edge and left != 0:
                                continue
                            # Can only use right edge if at last column
                            if use_right_edge and right != cols - 1:
                                continue

                            if is_valid_hollow_rectangle(grid, top, left, bottom, right, use_left_edge, use_right_edge):
                                # Determine interior bounds
                                interior_left = left if use_left_edge else left + 1
                                interior_right = right + 1 if use_right_edge else right

                                # Fill interior with 4s
                                for i in range(top + 1, bottom):
                                    for j in range(interior_left, interior_right):
                                        if (i, j) not in processed:
                                            result[i][j] = 4
                                            processed.add((i, j))

    return result
