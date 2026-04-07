def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing values 1, 2, and 8
    2. Output is a rectangular subgrid with border of 8s and interior containing 2s
    3. The grid contains multiple rectangular frames where all border cells are 8s
    4. We need to find rectangles of the same size and identify the minority pattern
    5. If all patterns are identical, apply a transformation to the interior

    Procedure:
    1. Iterate through all possible rectangle sizes (minimum 3x3)
    2. For each size, find all rectangles where all border cells are 8s
    3. Group rectangles by their content pattern using string representation
    4. Return the least common rectangle pattern among those of the same size
    5. If all rectangles are identical, apply diagonal transformation to 2s in interior
    """

    def is_complete_rectangle(grid, r, c, h, w):
        """Check if region starting at (r,c) with height h, width w forms a complete rectangle bounded by 8s"""
        if r + h > len(grid) or c + w > len(grid[0]):
            return False

        # Check top and bottom borders
        for col in range(c, c + w):
            if grid[r][col] != 8 or grid[r + h - 1][col] != 8:
                return False

        # Check left and right borders
        for row in range(r, r + h):
            if grid[row][c] != 8 or grid[row][c + w - 1] != 8:
                return False

        return True

    def extract_rectangle(grid, r, c, h, w):
        """Extract rectangle region"""
        rectangle = []
        for row in range(r, r + h):
            rectangle.append(grid[row][c : c + w])
        return rectangle

    # Find all complete rectangles
    rectangles = []
    rows, cols = len(grid), len(grid[0])

    # Try different rectangle sizes
    for h in range(3, min(rows, cols) + 1):  # minimum size 3x3
        for w in range(3, min(rows, cols) + 1):
            rects_of_this_size = []
            for r in range(rows - h + 1):
                for c in range(cols - w + 1):
                    if is_complete_rectangle(grid, r, c, h, w):
                        rect = extract_rectangle(grid, r, c, h, w)
                        rects_of_this_size.append(rect)

            # If we found multiple rectangles of the same size, use them
            if len(rects_of_this_size) >= 2:
                rectangles = rects_of_this_size
                break
        if rectangles:
            break

    assert len(rectangles) >= 2, f"Found {len(rectangles)} rectangles, need at least 2"

    # Find unique or minority rectangle patterns
    h, w = len(rectangles[0]), len(rectangles[0][0])

    # Count frequency of each rectangle pattern
    from collections import Counter

    rect_strings = [str(rect) for rect in rectangles]
    pattern_counts = Counter(rect_strings)

    # Find the least common rectangle
    min_count = min(pattern_counts.values())
    for rect, rect_str in zip(rectangles, rect_strings):
        if pattern_counts[rect_str] == min_count:
            # Use this rectangle as the base
            result = [row[:] for row in rect]
            break
    else:
        # Fallback: use first rectangle
        result = [row[:] for row in rectangles[0]]

    # If all rectangles are the same, apply transformation
    if len(pattern_counts) == 1:
        # All rectangles are identical - apply diagonal transformation to internal area
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if result[i][j] == 2:
                    # Convert some 2s to 8s based on position
                    if (i + j) % 2 == 1:  # odd positions become 8
                        result[i][j] = 8

    return result
