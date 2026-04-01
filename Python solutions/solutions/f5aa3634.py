def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing scattered patterns of various colors
    2. Output is a rectangular region extracted from the input
    3. The output pattern appears exactly twice in the input grid
    4. Example 1: 3x3 region [[0,8,0],[8,8,8],[5,5,5]] appears at (6,2) and (11,2)
    5. Example 2: 3x4 region [[0,5,8,8],[3,5,3,8],[0,3,3,0]] appears at (0,1) and (6,8)
    6. Example 3: 3x4 region [[0,0,5,9],[7,7,5,9],[0,5,7,0]] appears at (0,9) and (11,10)

    Procedure:
    1. Try different rectangular region sizes (prioritize 3xN patterns)
    2. For each possible region position, extract the pattern
    3. Count how many times this exact pattern appears in the grid
    4. Collect patterns that appear exactly twice and have meaningful content
    5. Return the pattern with highest non-zero count (most content)
    """

    rows, cols = len(grid), len(grid[0])

    # Function to extract a rectangular region
    def get_region(r, c, h, w):
        if r + h > rows or c + w > cols:
            return None
        region = []
        for i in range(h):
            row = []
            for j in range(w):
                row.append(grid[r + i][c + j])
            region.append(row)
        return region

    # Function to count occurrences of a pattern
    def count_pattern(pattern):
        h, w = len(pattern), len(pattern[0])
        count = 0
        positions = []

        for r in range(rows - h + 1):
            for c in range(cols - w + 1):
                match = True
                for i in range(h):
                    for j in range(w):
                        if grid[r + i][c + j] != pattern[i][j]:
                            match = False
                            break
                    if not match:
                        break

                if match:
                    count += 1
                    positions.append((r, c))

        return count, positions

    # Collect all patterns that appear exactly twice
    duplicate_patterns = []

    # Try specific sizes that appeared in training examples first (smaller first)
    priority_sizes = [(3, 3), (3, 4)]

    for height, width in priority_sizes:
        for r in range(rows - height + 1):
            for c in range(cols - width + 1):
                pattern = get_region(r, c, height, width)
                if pattern is None:
                    continue

                # Check if pattern has any non-zero content
                has_content = any(val != 0 for row in pattern for val in row)
                if not has_content:
                    continue

                # Count how many times this pattern appears
                count, positions = count_pattern(pattern)

                # If exactly two occurrences, collect it
                if count == 2:
                    non_zero_count = sum(
                        1 for row in pattern for val in row if val != 0
                    )
                    size = height * width
                    # Prioritize patterns with meaningful content
                    if non_zero_count >= 3:
                        duplicate_patterns.append((non_zero_count, size, pattern))

    # If no priority patterns found, try other 3-row patterns
    if not duplicate_patterns:
        for width in range(3, cols + 1):
            if (3, width) in priority_sizes:  # Already tried
                continue
            for r in range(rows - 3 + 1):
                for c in range(cols - width + 1):
                    pattern = get_region(r, c, 3, width)
                    if pattern is None:
                        continue

                    # Check if pattern has any non-zero content
                    has_content = any(val != 0 for row in pattern for val in row)
                    if not has_content:
                        continue

                    # Count how many times this pattern appears
                    count, positions = count_pattern(pattern)

                    # If exactly two occurrences, collect it
                    if count == 2:
                        non_zero_count = sum(
                            1 for row in pattern for val in row if val != 0
                        )
                        size = 3 * width
                        # For 3-row patterns, be more lenient with density
                        if non_zero_count >= 3:
                            duplicate_patterns.append((non_zero_count, size, pattern))

    # If no 3-row patterns found, try other sizes
    if not duplicate_patterns:
        for height in range(1, rows + 1):
            if height == 3:  # Already tried
                continue
            for width in range(1, cols + 1):
                for r in range(rows - height + 1):
                    for c in range(cols - width + 1):
                        pattern = get_region(r, c, height, width)
                        if pattern is None:
                            continue

                        # Check if pattern has any non-zero content
                        has_content = any(val != 0 for row in pattern for val in row)
                        if not has_content:
                            continue

                        # Count how many times this pattern appears
                        count, positions = count_pattern(pattern)

                        # If exactly two occurrences, collect it
                        if count == 2:
                            non_zero_count = sum(
                                1 for row in pattern for val in row if val != 0
                            )
                            size = height * width
                            density = non_zero_count / size

                            # Only consider patterns with reasonable density and meaningful content
                            if non_zero_count >= 3 and density >= 0.25:
                                duplicate_patterns.append(
                                    (non_zero_count, size, pattern)
                                )

    # Return the pattern with most content and smallest size (prioritize compact patterns)
    if duplicate_patterns:
        # Sort by non-zero count (descending), then by size (ascending) to prefer compact patterns
        duplicate_patterns.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return duplicate_patterns[0][2]

    return [[0]]
