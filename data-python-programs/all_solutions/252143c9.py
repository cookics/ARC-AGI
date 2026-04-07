def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a square grid with scattered non-background colored cells
    2. Output is same size grid with background color and a single diagonal line of one color
    3. The diagonal line is an anti-diagonal (r + c = constant)
    4. The line color appears most frequently in the input among non-background colors
    5. The line extends from first occurrence to either last occurrence or end of diagonal

    Procedure:
    1. Find background color (most frequent color in grid)
    2. Iterate through all anti-diagonals (r + c = constant)
    3. For each anti-diagonal, count occurrences of each non-background color
    4. For each color, determine segment: first to last occurrence vs first to end
    5. Select color and segment with highest count (tiebreak by higher color value)
    6. Create output grid filled with background, mark segment with chosen color
    """

    rows, cols = len(grid), len(grid[0])

    # Find background color (most frequent)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1

    background = max(color_count, key=color_count.get)

    best_score = 0
    best_segment = []
    best_color = background

    # Only try anti-diagonals (r + c = constant)
    for sum_val in range(rows + cols):
        diagonal = []
        for r in range(rows):
            c = sum_val - r
            if 0 <= c < cols:
                diagonal.append((r, c))

        if not diagonal:
            continue

        # Count non-background colors and track positions
        color_counts = {}
        color_positions = {}
        for i, (r, c) in enumerate(diagonal):
            color = grid[r][c]
            if color != background:
                color_counts[color] = color_counts.get(color, 0) + 1
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append(i)

        # For each color, choose between "first to last" or "first to end" based on length difference
        for color, count in color_counts.items():
            positions = color_positions[color]
            first_pos = min(positions)
            last_pos = max(positions)

            # Option 1: first to last occurrence
            segment1 = diagonal[first_pos : last_pos + 1]

            # Option 2: first to end of diagonal
            segment2 = diagonal[first_pos:]

            # If the difference in length is small, prefer "first to end"
            # If the difference is large, prefer "first to last"
            if len(segment2) - len(segment1) <= 2:  # Small difference threshold
                segment = segment2
            else:
                segment = segment1

            # Check if this beats the current best (prioritize higher count, then higher color)
            if count > best_score or (count == best_score and color > best_color):
                best_score = count
                best_segment = segment
                best_color = color

    # Create output
    result = [[background for _ in range(cols)] for _ in range(rows)]
    for r, c in best_segment:
        result[r][c] = best_color

    return result
