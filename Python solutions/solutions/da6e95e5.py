def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with background color and multiple scattered rectangular patterns
    2. Different colored patterns form frames/structures that repeat
    3. Output is a specific rectangular pattern that appears exactly 2 times
    4. The pattern has specific dimensions (3x3, 2x2, or 4x5) and contains mix of background and non-background

    Procedure:
    1. Find the background color
    2. Try specific target sizes in priority order
    3. For each size, find all patterns that appear exactly 2 times
    4. Return the first matching pattern with good non-background ratio
    """

    rows, cols = len(grid), len(grid[0])

    # Find background color (most frequent)
    color_counts = {}
    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            color_counts[color] = color_counts.get(color, 0) + 1

    background_color = max(color_counts, key=color_counts.get)

    # Try sizes in priority order
    priority_sizes = [(3, 3), (2, 2), (4, 5), (5, 4), (3, 4), (4, 3), (3, 5), (5, 3)]

    for h, w in priority_sizes:
        pattern_counts = {}
        first_pattern = {}

        # Extract all rectangles of this size
        for r in range(rows - h + 1):
            for c in range(cols - w + 1):
                pattern = []
                has_non_background = False
                non_bg_count = 0

                for i in range(h):
                    row = []
                    for j in range(w):
                        cell_value = grid[r + i][c + j]
                        row.append(cell_value)
                        if cell_value != background_color:
                            has_non_background = True
                            non_bg_count += 1
                    pattern.append(tuple(row))
                pattern = tuple(pattern)

                if has_non_background:
                    if pattern not in pattern_counts:
                        pattern_counts[pattern] = 0
                        first_pattern[pattern] = (non_bg_count, h * w)
                    pattern_counts[pattern] += 1

        # Find ALL patterns that appear exactly 2 times and pick the best one
        candidates = []
        for pattern, count in pattern_counts.items():
            if count == 2:
                non_bg_count, area = first_pattern[pattern]
                ratio = non_bg_count / area

                # Check if ratio is reasonable (not too sparse)
                if ratio >= 0.25:
                    # Count ALL colors in this pattern
                    all_colors = set()
                    non_bg_colors = set()
                    for row in pattern:
                        for cell in row:
                            all_colors.add(cell)
                            if cell != background_color:
                                non_bg_colors.add(cell)

                    # Only accept patterns with exactly ONE non-background color
                    if len(non_bg_colors) == 1:
                        # Score by non-background count (more is better)
                        score = non_bg_count
                        candidates.append((score, pattern))

        # Return the pattern with the most non-background cells
        if candidates:
            candidates.sort(reverse=True)
            best_pattern = candidates[0][1]
            result = []
            for row in best_pattern:
                result.append(list(row))
            return result

    # Fallback: try any small size with exactly 2 occurrences
    for h in range(2, 8):
        for w in range(2, 8):
            if (h, w) in priority_sizes:
                continue  # Already tried

            pattern_counts = {}
            first_pattern = {}

            for r in range(rows - h + 1):
                for c in range(cols - w + 1):
                    pattern = []
                    has_non_background = False
                    non_bg_count = 0

                    for i in range(h):
                        row = []
                        for j in range(w):
                            cell_value = grid[r + i][c + j]
                            row.append(cell_value)
                            if cell_value != background_color:
                                has_non_background = True
                                non_bg_count += 1
                        pattern.append(tuple(row))
                    pattern = tuple(pattern)

                    if has_non_background:
                        if pattern not in pattern_counts:
                            pattern_counts[pattern] = 0
                            first_pattern[pattern] = (non_bg_count, h * w)
                        pattern_counts[pattern] += 1

            for pattern, count in pattern_counts.items():
                if count == 2:
                    non_bg_count, area = first_pattern[pattern]
                    ratio = non_bg_count / area
                    if ratio >= 0.25:
                        result = []
                        for row in pattern:
                            result.append(list(row))
                        return result

    return [[background_color]]
