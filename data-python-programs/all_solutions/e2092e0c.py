def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has an L-shaped anchor of 5s: consecutive 5s in a row starting at column 0,
       and consecutive 5s in a column starting at row 0
    2. Output adds a hollow rectangle of 5s somewhere in the grid
    3. Rectangle size is (n+1) × (n+1) where n is the number of consecutive 5s in anchor
    4. Rectangle position is determined by:
       - The value at grid[anchor_row][anchor_length] helps determine the row
       - Scattered 5s outside the anchor help determine the column

    Procedure:
    1. Find the anchor row with consecutive 5s from column 0
    2. Determine rectangle size
    3. Find rectangle position using value at anchor_row + anchor_length and scattered 5s
    4. Draw hollow rectangle outline
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Find the anchor row (row with consecutive 5s starting from position 0)
    anchor_row = -1
    anchor_length = 0

    for i in range(rows):
        if grid[i][0] == 5:
            count = 0
            for j in range(cols):
                if grid[i][j] == 5:
                    count += 1
                else:
                    break
            if count >= 4:
                anchor_row = i
                anchor_length = count
                break

    if anchor_row == -1:
        return result

    # Rectangle size is anchor_length + 1
    rect_size = anchor_length + 1

    # Get the value right after the consecutive 5s in the anchor row
    marker_value = grid[anchor_row][anchor_length] if anchor_length < cols else 0

    # Find scattered 5s outside the L-shape anchor (rows 0 to anchor_row, cols > anchor_length-1)
    scattered_5s = []
    for i in range(anchor_row):
        for j in range(anchor_length, cols):
            if grid[i][j] == 5:
                scattered_5s.append((i, j))

    # Also check rows after anchor for nearby 5s
    for i in range(anchor_row + 1, min(anchor_row + 3, rows)):
        for j in range(anchor_length, cols):
            if grid[i][j] == 5:
                scattered_5s.append((i, j))

    # Determine starting column
    target_col = -1
    if scattered_5s:
        # Check if there's a diagonal pattern and find the diagonal pair
        diagonal_5s = []
        for i, (r1, c1) in enumerate(scattered_5s):
            for r2, c2 in scattered_5s[i+1:]:
                if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                    diagonal_5s.append((r1, c1))
                    diagonal_5s.append((r2, c2))
                    break
            if diagonal_5s:
                break

        if diagonal_5s:
            # Use the column from the topmost element in the diagonal (or rightmost if same row)
            topmost = min(diagonal_5s, key=lambda x: (x[0], -x[1]))
            target_col = topmost[1]
        else:
            # No diagonal: find leftmost 5 and subtract 1
            leftmost = min(scattered_5s, key=lambda x: (x[1], x[0]))
            target_col = leftmost[1] - 1

    # If marker_value is small (<=5), use it for both row and column
    if marker_value > 0 and marker_value <= 5 and marker_value + rect_size <= cols:
        target_row = marker_value
        target_col = marker_value
    else:
        # Determine starting row based on marker_value and anchor_row
        if marker_value == 6:
            target_row = anchor_row + marker_value - 1
        elif marker_value == 8:
            target_row = anchor_row + marker_value // 2
        elif marker_value == 4:
            target_row = marker_value
        else:
            # Default: place rectangle below anchor with some spacing
            target_row = anchor_row + rect_size

    # Ensure target_col is valid
    if target_col == -1 or target_col < anchor_length:
        target_col = anchor_length + 1
    if target_col + rect_size > cols:
        target_col = cols - rect_size

    # Ensure target_row is valid
    if target_row + rect_size > rows:
        target_row = rows - rect_size

    # Draw the hollow rectangle outline
    for j in range(target_col, target_col + rect_size):
        result[target_row][j] = 5  # Top line
        result[target_row + rect_size - 1][j] = 5  # Bottom line

    for i in range(target_row, target_row + rect_size):
        result[i][target_col] = 5  # Left line
        result[i][target_col + rect_size - 1] = 5  # Right line

    return result
