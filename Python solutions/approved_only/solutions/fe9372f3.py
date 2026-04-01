def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing a plus-shaped pattern of 2s (center cell with 2s above, below, left, and right).
    2. Output preserves the original plus pattern of 2s in their exact positions.
    3. Output draws horizontal and vertical lines of 8s through the center of the plus pattern.
    4. Output draws diagonal lines of 1s extending from the center in all four diagonal directions.
    5. Output replaces some 8s with 4s at regular intervals (at distance 4 from center, then every 3 positions after that).

    Procedure:
    1. Find the center of the plus pattern by locating a 2 that has adjacent 2s in all four cardinal directions.
    2. Create output grid as a copy of the input grid.
    3. Draw horizontal and vertical lines of 8s through the center row and column (preserving existing 2s).
    4. Draw diagonal lines of 1s extending from center in all four diagonal directions (preserving existing 2s).
    5. Replace specific 8s with 4s at calculated intervals along the horizontal and vertical lines.
    """

    rows, cols = len(grid), len(grid[0])

    # Find the center of the plus pattern
    center_row, center_col = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                # Check if this is the center of a plus pattern
                if (
                    r > 0
                    and r < rows - 1
                    and c > 0
                    and c < cols - 1
                    and grid[r - 1][c] == 2
                    and grid[r + 1][c] == 2
                    and grid[r][c - 1] == 2
                    and grid[r][c + 1] == 2
                ):
                    center_row, center_col = r, c
                    break
        if center_row is not None:
            break

    assert center_row is not None and center_col is not None, (
        "Could not find plus pattern center"
    )

    # Create output grid
    result = [row[:] for row in grid]

    # Draw horizontal line of 8s through center
    for c in range(cols):
        if result[center_row][c] != 2:
            result[center_row][c] = 8

    # Draw vertical line of 8s through center
    for r in range(rows):
        if result[r][center_col] != 2:
            result[r][center_col] = 8

    # Draw diagonal lines of 1s
    # Main diagonal (top-left to bottom-right)
    for i in range(max(rows, cols)):
        # Upper-left diagonal
        r1, c1 = center_row - i, center_col - i
        if 0 <= r1 < rows and 0 <= c1 < cols and result[r1][c1] != 2:
            result[r1][c1] = 1

        # Lower-right diagonal
        r2, c2 = center_row + i, center_col + i
        if 0 <= r2 < rows and 0 <= c2 < cols and result[r2][c2] != 2:
            result[r2][c2] = 1

    # Anti-diagonal (top-right to bottom-left)
    for i in range(max(rows, cols)):
        # Upper-right diagonal
        r1, c1 = center_row - i, center_col + i
        if 0 <= r1 < rows and 0 <= c1 < cols and result[r1][c1] != 2:
            result[r1][c1] = 1

        # Lower-left diagonal
        r2, c2 = center_row + i, center_col - i
        if 0 <= r2 < rows and 0 <= c2 < cols and result[r2][c2] != 2:
            result[r2][c2] = 1

    # Add 4s at regular intervals
    # Pattern: 4s at distance 4 from center, then every 3 positions after that

    # Horizontal line 4s
    for c in range(cols):
        if result[center_row][c] == 8:
            dist = c - center_col
            if dist > 0:  # Right side of center
                if dist == 4 or (dist > 4 and (dist - 4) % 3 == 0):
                    result[center_row][c] = 4
            elif dist < 0:  # Left side of center
                abs_dist = abs(dist)
                if abs_dist == 4 or (abs_dist > 4 and (abs_dist - 4) % 3 == 0):
                    result[center_row][c] = 4

    # Vertical line 4s
    for r in range(rows):
        if result[r][center_col] == 8:
            dist = r - center_row
            if dist > 0:  # Below center
                if dist == 4 or (dist > 4 and (dist - 4) % 3 == 0):
                    result[r][center_col] = 4
            elif dist < 0:  # Above center
                abs_dist = abs(dist)
                if abs_dist == 4 or (abs_dist > 4 and (abs_dist - 4) % 3 == 0):
                    result[r][center_col] = 4

    return result
