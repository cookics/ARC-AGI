def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The input has 7s everywhere except some 5s positioned on diagonal lines.
    The 5s get transformed to either 2s or 8s based on whether they form continuous
    diagonal lines and the position of the main diagonal line.

    Pattern discovered:
    1. Find continuous diagonal lines going down-left (row+1, col-1)
    2. If the main diagonal line passes through center (sum=15), then:
       diagonal lines → 2s, others → 8s
    3. If main diagonal lines are away from center, then:
       diagonal lines → 8s, others → 2s

    Procedure:
    1. Find all positions with value 5
    2. Identify continuous diagonal lines
    3. Determine if main diagonal is at center or not
    4. Apply appropriate transformation
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])
    center_sum = rows - 1  # For 16x16 grid, center anti-diagonal sum is 15

    # Find all positions with value 5
    fives = set()
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 5:
                fives.add((i, j))

    if not fives:
        return result

    # Find continuous diagonal lines and identify which ones are main
    def find_diagonal_lines():
        visited = set()
        diagonal_lines = []

        for r, c in fives:
            if (r, c) in visited:
                continue

            # Find the start of diagonal line passing through this point
            start_r, start_c = r, c
            while (start_r - 1, start_c + 1) in fives:
                start_r -= 1
                start_c += 1

            # Collect all points in this diagonal line
            line = []
            curr_r, curr_c = start_r, start_c
            while (curr_r, curr_c) in fives:
                line.append((curr_r, curr_c))
                visited.add((curr_r, curr_c))
                curr_r += 1
                curr_c -= 1

            # Only consider lines with 3+ points as diagonal lines
            if len(line) >= 3:
                diagonal_lines.append(line)

        return diagonal_lines

    diagonal_lines = find_diagonal_lines()
    diagonal_points = set()

    # Collect all points that are part of diagonal lines
    for line in diagonal_lines:
        for point in line:
            diagonal_points.add(point)

    # Check if any major diagonal line passes through center
    has_center_diagonal = False
    for line in diagonal_lines:
        if len(line) >= 5:  # Consider long lines as major
            for r, c in line:
                if r + c == center_sum:
                    has_center_diagonal = True
                    break
            if has_center_diagonal:
                break

    # Apply transformations based on pattern
    for r, c in fives:
        is_diagonal = (r, c) in diagonal_points

        if has_center_diagonal:
            # Main diagonal through center: diagonal lines → 2s, others → 8s
            if is_diagonal:
                result[r][c] = 2
            else:
                result[r][c] = 8
        else:
            # Main diagonals away from center: diagonal lines → 8s, others → 2s
            if is_diagonal:
                result[r][c] = 8
            else:
                result[r][c] = 2

    return result
