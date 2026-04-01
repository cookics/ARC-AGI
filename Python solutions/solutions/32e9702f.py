def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with 0s as background and non-zero values as colored objects.
    2. Output grid has all background cells (0s) transformed to 5s.
    3. Horizontal segments of colored cells undergo position transformations.
    4. If a colored segment starts at column 0, its rightmost cell is removed.
    5. If a colored segment doesn't start at column 0, the entire segment shifts left by 1 position.

    Procedure:
    1. Create result grid by replacing all 0s with 5s while preserving other values.
    2. For each row, identify all horizontal segments of colored cells (non-zero values).
    3. Clear the original colored cell positions in the result grid to background (5).
    4. Apply transformation rules to each segment based on its starting position.
    5. Place the transformed segments in their new positions in the result grid.
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [[5 if cell == 0 else cell for cell in row] for row in grid]

    for r in range(rows):
        # Find horizontal segments of colored cells (non-zero, non-5)
        segments = []
        start = None

        for c in range(cols):
            if grid[r][c] != 0:  # colored cell
                if start is None:
                    start = c
            else:  # background cell (0)
                if start is not None:
                    segments.append((start, c - 1, grid[r][start]))
                    start = None

        # Handle segment that extends to the end
        if start is not None:
            segments.append((start, cols - 1, grid[r][start]))

        # Clear the row first (set to background)
        for c in range(cols):
            if grid[r][c] != 0:
                result[r][c] = 5

        # Apply transformation to each segment
        for start_col, end_col, color in segments:
            segment_length = end_col - start_col + 1

            if start_col == 0:
                # Segment starts at column 0: remove rightmost cell
                for i in range(segment_length - 1):
                    result[r][i] = color
            else:
                # Segment doesn't start at column 0: shift left by 1
                new_start = start_col - 1
                for i in range(segment_length):
                    if new_start + i < cols:
                        result[r][new_start + i] = color

    return result
