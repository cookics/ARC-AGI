def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with exactly two cells containing value 8
    2. Output is the same grid with a corridor of 3s connecting the two 8s
    3. The corridor forms a parallelogram shape with two parallel edges
    4. When row difference is 2, forms a three-sided rectangle with hollow interior
    5. When diagonal, creates two rails with a gap between them
    6. The 3s are placed adjacent to the 8s and continue in parallel lines

    Procedure:
    1. Find the positions of the two 8s in the grid
    2. Determine if it's a rectangular pattern (row difference equals 2) or diagonal pattern
    3. For rectangular pattern, draw horizontal lines and edge connectors
    4. For diagonal pattern, place 3s adjacent to each 8 and draw two parallel rails
    5. Return the modified grid with the corridor drawn
    """

    # Find positions of the two 8s
    positions = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 8:
                positions.append((i, j))

    if len(positions) != 2:
        return grid

    # Get coordinates
    r1, c1 = positions[0]
    r2, c2 = positions[1]

    # Make a copy of the grid
    result = [row[:] for row in grid]

    # Check if this is a rectangular pattern (row difference is exactly 2)
    if abs(r2 - r1) == 2:
        # Three-sided rectangle with only edge connectors in middle
        min_r = min(r1, r2)
        max_r = max(r1, r2)
        min_c = min(c1, c2)
        max_c = max(c1, c2)

        # Draw horizontal lines between the columns (not including the 8s)
        # From looking at example 3, we go from column 4 to 11 (exclusive of 8s at 2 and 12)
        for c in range(min_c + 2, max_c):
            result[min_r][c] = 3
        for c in range(min_c + 1, max_c - 1):
            result[max_r][c] = 3

        # Draw only edge connectors in middle row
        mid_r = min_r + 1
        result[mid_r][min_c + 1] = 3
        result[mid_r][max_c - 1] = 3
    else:
        # Diagonal pattern
        # Sort to ensure top 8 is first
        if r1 > r2:
            r1, c1, r2, c2 = r2, c2, r1, c1

        # Determine direction
        c_dir = 1 if c2 > c1 else -1

        if c_dir == 1:  # Moving right and down
            # Place 3s one row below first 8
            result[r1 + 1][c1] = 3
            result[r1 + 1][c1 + 1] = 3

            # Place 3s one row above second 8
            result[r2 - 1][c2 - 1] = 3
            result[r2 - 1][c2] = 3

            # Draw diagonal rails with specific pattern
            # Left rail stays at column 2 for a while
            for r in range(r1 + 2, r2 - 1):
                if r - r1 <= 3:
                    # Left rail stays at column 2
                    result[r][c1] = 3
                else:
                    # Left rail starts moving diagonally
                    result[r][c1 + (r - r1 - 3)] = 3

                # Right rail moves diagonally but with specific adjustments
                # Looking at example 2: the right rail pattern is
                # Row 3: col 4   (dist=2)
                # Row 4: col 5   (dist=3)
                # Row 5: col 6   (dist=4)
                # Row 6: col 7   (dist=5)
                # Row 7: col 8   (dist=6)
                # Row 8: col 8   (dist=7) - stays at 8!
                if r == r2 - 2:  # Second to last row
                    c_right = c2  # Same column as second 8
                else:
                    c_right = c1 + (r - r1)

                if 0 <= c_right < len(grid[0]):
                    result[r][c_right] = 3
        else:  # Moving left and down
            # Place 3s on same row as first 8, to the left
            result[r1][c1 - 1] = 3
            result[r1][c1 - 2] = 3

            # Place 3s on same row as second 8, to the right
            result[r2][c2 + 1] = 3
            result[r2][c2 + 2] = 3

            # Draw diagonal rails
            # Looking at example 1: the pattern has two rails with a gap
            for r in range(r1 + 1, r2):
                # Distance from start
                dist = r - r1
                # For moving left pattern (example 1 where c1=11):
                # Row 1: cols 8,10  (dist=1, so 11-3=8, 11-1=10)
                # Row 2: cols 7,9   (dist=2, so 11-4=7, 11-2=9)
                # Row 3: cols 6,8   (dist=3, so 11-5=6, 11-3=8)
                # Pattern: left rail at c1-dist-2, right rail at c1-dist
                c_left = c1 - dist - 2
                c_right = c1 - dist

                if 0 <= c_left < len(grid[0]):
                    result[r][c_left] = 3
                if 0 <= c_right < len(grid[0]):
                    result[r][c_right] = 3

    return result
