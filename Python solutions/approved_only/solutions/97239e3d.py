def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 17x17 grid containing values 0, 8, and special values (non-0, non-8).
    2. Output is the same grid with certain 0s transformed to special values.
    3. The grid has separator rows/columns at positions 0, 4, 8, 12, 16 that divide it into blocks.
    4. Special values appear scattered across the grid at various positions.
    5. For each distinct special value, a rectangular region is created that spans from the separators that bound all positions of that value.
    6. Within each rectangular region, specific 0s are transformed to the special value on boundaries and at center positions of 3x3 blocks.

    Procedure:
    1. Find all positions of each special value in the grid.
    2. For each special value, determine the bounding separators that contain all its positions.
    3. Create rectangular regions based on these bounding separators.
    4. Apply transformations within each region by changing 0s to the special value at boundary positions and center positions of 3x3 blocks.
    """

    # Copy the grid
    result = [row[:] for row in grid]

    # Find all special values (not 0 or 8)
    special_values = {}
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            val = grid[r][c]
            if val != 0 and val != 8:
                if val not in special_values:
                    special_values[val] = []
                special_values[val].append((r, c))

    # Separator positions
    separators = [0, 4, 8, 12, 16]

    def find_bounding_separators(positions):
        """Find the range of separators that bound the given positions"""
        min_pos = min(positions)
        max_pos = max(positions)

        # Find the separator that covers the minimum position
        min_sep = 0
        for sep in separators:
            if sep <= min_pos:
                min_sep = sep

        # Find the separator that covers the maximum position
        max_sep = 16  # Default to grid boundary
        for sep in separators:
            if sep >= max_pos:
                max_sep = sep
                break

        return min_sep, max_sep

    # Process each special value
    for special_val, positions in special_values.items():
        # Get row and column positions
        rows = [pos[0] for pos in positions]
        cols = [pos[1] for pos in positions]

        # Find bounding separators
        min_row, max_row = find_bounding_separators(rows)
        min_col, max_col = find_bounding_separators(cols)

        # Apply transformation within the region
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                if grid[r][c] == 0:
                    # Transform if it's on the boundary of the region
                    if r == min_row or r == max_row or c == min_col or c == max_col:
                        result[r][c] = special_val
                    # Transform if it's the center of a 3x3 block (row%4==2 and col%4==2)
                    elif r % 4 == 2 and c % 4 == 2:
                        result[r][c] = special_val

    return result
