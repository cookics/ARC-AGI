def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a 3x3 pattern made of 1s and 8s, plus scattered 4s
    2. Output is a grid with the 3x3 pattern rotated and a single 4 placed
    3. The 3x3 pattern is rotated based on its structural properties:
       - If it has duplicate rows: rotate 180 degrees
       - If it has duplicate columns: rotate 90 degrees clockwise
       - Otherwise: rotate 90 degrees counterclockwise
    4. The single output 4 is placed at offset (±2, ±2) from the pattern center
    5. The 4's position is in the quadrant that had the most input 4s

    Procedure:
    1. Locate the 3x3 pattern of 1s and 8s in the input grid
    2. Extract the pattern and determine rotation type based on duplicate rows/columns
    3. Apply the appropriate rotation transformation to the pattern
    4. Count the number of 4s in each quadrant relative to the pattern center
    5. Create output grid and place the rotated pattern at the same position
    6. Place a single 4 at offset (±2, ±2) in the quadrant with most input 4s
    """

    def has_pattern(r, c):
        # Check if there's a 3x3 pattern of 1s and 8s starting at (r,c)
        values = set()
        for dr in range(3):
            for dc in range(3):
                val = grid[r + dr][c + dc]
                if val not in [1, 8]:
                    return False
                values.add(val)
        return 1 in values and 8 in values

    def find_pattern():
        # Find a 3x3 region containing 1s and 8s
        for r in range(len(grid) - 2):
            for c in range(len(grid[0]) - 2):
                if has_pattern(r, c):
                    return r, c
        return None, None

    def extract_pattern(r, c):
        return [[grid[r + dr][c + dc] for dc in range(3)] for dr in range(3)]

    def rotate_90_ccw(pattern):
        # Rotate 90 degrees counterclockwise
        n = len(pattern)
        result = [[0 for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for c in range(n):
                result[n - 1 - c][r] = pattern[r][c]
        return result

    def rotate_90_cw(pattern):
        # Rotate 90 degrees clockwise
        n = len(pattern)
        result = [[0 for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for c in range(n):
                result[c][n - 1 - r] = pattern[r][c]
        return result

    def rotate_180(pattern):
        # Rotate 180 degrees
        n = len(pattern)
        result = [[0 for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for c in range(n):
                result[n - 1 - r][n - 1 - c] = pattern[r][c]
        return result

    def has_duplicate_rows(pattern):
        # Check if the pattern has any duplicate rows
        for i in range(len(pattern)):
            for j in range(i + 1, len(pattern)):
                if pattern[i] == pattern[j]:
                    return True
        return False

    def has_duplicate_columns(pattern):
        # Check if the pattern has any duplicate columns
        n = len(pattern)
        for c1 in range(n):
            for c2 in range(c1 + 1, n):
                col1 = [pattern[r][c1] for r in range(n)]
                col2 = [pattern[r][c2] for r in range(n)]
                if col1 == col2:
                    return True
        return False

    def find_dominant_quadrant(center_r, center_c):
        quadrant_counts = [0, 0, 0, 0]  # top-left, top-right, bottom-left, bottom-right

        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if grid[r][c] == 4:
                    if r < center_r and c < center_c:
                        quadrant_counts[0] += 1  # top-left
                    elif r < center_r and c > center_c:
                        quadrant_counts[1] += 1  # top-right
                    elif r > center_r and c < center_c:
                        quadrant_counts[2] += 1  # bottom-left
                    elif r > center_r and c > center_c:
                        quadrant_counts[3] += 1  # bottom-right

        return quadrant_counts.index(max(quadrant_counts))

    def place_pattern(result, pattern, r, c):
        for dr in range(3):
            for dc in range(3):
                result[r + dr][c + dc] = pattern[dr][dc]

    def place_4(result, center_r, center_c, quadrant):
        offsets = [
            (-2, -2),
            (-2, 2),
            (2, -2),
            (2, 2),
        ]  # top-left, top-right, bottom-left, bottom-right
        dr, dc = offsets[quadrant]
        result[center_r + dr][center_c + dc] = 4

    # Find the 3x3 pattern of 1s and 8s
    pattern_top_row, pattern_left_col = find_pattern()

    # Extract and transform the 3x3 pattern
    pattern = extract_pattern(pattern_top_row, pattern_left_col)

    # Determine which transformation to use based on pattern characteristics
    if has_duplicate_rows(pattern):
        rotated_pattern = rotate_180(pattern)
    elif has_duplicate_columns(pattern):
        rotated_pattern = rotate_90_cw(pattern)
    else:
        rotated_pattern = rotate_90_ccw(pattern)

    # Find which quadrant has the most 4s
    center_r, center_c = pattern_top_row + 1, pattern_left_col + 1
    quadrant = find_dominant_quadrant(center_r, center_c)

    # Create output grid
    result = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]

    # Place rotated pattern
    place_pattern(result, rotated_pattern, pattern_top_row, pattern_left_col)

    # Place the 4 in the dominant quadrant
    place_4(result, center_r, center_c, quadrant)

    return result
