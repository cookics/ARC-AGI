def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a background color (most frequent), marker 5s, pattern colors, and noise
    2. Output is vertically stacked 6×6 hollow diamond patterns, one per pattern color
    3. Pattern colors are non-background, non-5 colors with sufficient occurrences
    4. Hollow diamond shape has the pattern color on edges in a diamond formation

    Procedure:
    1. Find background color (most frequent)
    2. Count all non-background, non-5 colors
    3. Filter pattern colors (count > threshold, e.g., 5)
    4. Sort pattern colors numerically
    5. Reorder: for 3 colors [a,b,c] → [b,c,a]; for 2 colors [a,b] → [b,a]
    6. For each pattern color, generate 6×6 hollow diamond with that color
    7. Stack all diamonds vertically
    """
    from collections import Counter

    # Flatten grid and find background (most frequent color)
    all_cells = [cell for row in grid for cell in row]
    color_counts = Counter(all_cells)
    background = color_counts.most_common(1)[0][0]

    # Count non-background, non-5 colors
    pattern_counts = Counter()
    for cell in all_cells:
        if cell != background and cell != 5:
            pattern_counts[cell] += 1

    # Filter pattern colors (threshold = 5 occurrences)
    pattern_colors = [color for color, count in pattern_counts.items() if count > 5]

    # Sort pattern colors
    pattern_colors.sort()

    # Reorder based on number of colors
    if len(pattern_colors) == 3:
        # [a, b, c] → [b, c, a] (middle, highest, lowest)
        ordered_colors = [pattern_colors[1], pattern_colors[2], pattern_colors[0]]
    elif len(pattern_colors) == 2:
        # [a, b] → [b, a] (highest, lowest)
        ordered_colors = [pattern_colors[1], pattern_colors[0]]
    else:
        # Keep as is
        ordered_colors = pattern_colors

    # Create 6×6 hollow diamond pattern
    def create_diamond(color, bg):
        """
        Pattern:
        ..XX..
        .X..X.
        X....X
        X....X
        .X..X.
        ..XX..
        """
        diamond = []
        # Row 0: ..XX..
        diamond.append([bg, bg, color, color, bg, bg])
        # Row 1: .X..X.
        diamond.append([bg, color, bg, bg, color, bg])
        # Row 2: X....X
        diamond.append([color, bg, bg, bg, bg, color])
        # Row 3: X....X
        diamond.append([color, bg, bg, bg, bg, color])
        # Row 4: .X..X.
        diamond.append([bg, color, bg, bg, color, bg])
        # Row 5: ..XX..
        diamond.append([bg, bg, color, color, bg, bg])
        return diamond

    # Generate output by stacking diamonds
    result = []
    for color in ordered_colors:
        diamond = create_diamond(color, background)
        result.extend(diamond)

    return result
