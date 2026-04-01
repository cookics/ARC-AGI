def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains multiple colored patterns scattered on a background
    2. Background is the most frequent color
    3. Each non-background color forms a pattern within its bounding box
    4. Output is created by extracting and overlaying all patterns centrally
    5. Patterns are sorted by size (largest first) and overlaid in that order
    6. Output size equals the maximum dimension among all patterns
    7. Smaller patterns overwrite larger ones when overlaid

    Procedure:
    1. Identify background color (most frequent)
    2. Extract each non-background color's pattern and bounding box
    3. Determine output size from the largest pattern dimension
    4. Sort patterns by size descending
    5. Overlay patterns from largest to smallest, centering each one
    """
    from collections import Counter

    rows = len(grid)
    cols = len(grid[0])

    # Find background color (most frequent)
    color_counts = Counter()
    for row in grid:
        for cell in row:
            color_counts[cell] += 1

    background = color_counts.most_common(1)[0][0]

    # Extract each color's pattern
    color_cells = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background:
                color = grid[r][c]
                if color not in color_cells:
                    color_cells[color] = []
                color_cells[color].append((r, c))

    if not color_cells:
        return [[background]]

    # Extract pattern for each color
    patterns = []
    for color, cells in color_cells.items():
        # Find bounding box
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        height = max_r - min_r + 1
        width = max_c - min_c + 1

        # Get relative positions within bounding box
        rel_positions = set()
        for r, c in cells:
            rel_positions.add((r - min_r, c - min_c))

        # Make pattern square (pad to max dimension)
        max_dim = max(height, width)

        # Determine if we need to reverse based on edge density
        # If pattern is narrow and rightmost column has more cells than leftmost, reverse
        if height > width * 1.5:  # Significantly taller than wide
            # Count cells in leftmost vs rightmost column
            leftmost_count = sum(1 for r, c in rel_positions if c == 0)
            rightmost_count = sum(1 for r, c in rel_positions if c == width - 1)

            # If rightmost has more cells, reverse so it becomes the outer edge
            if rightmost_count > leftmost_count:
                rel_positions_adj = set()
                for r, c in rel_positions:
                    rel_positions_adj.add((r, width - 1 - c))
                rel_positions = rel_positions_adj

        # If pattern is wide and bottom row has more cells than top, reverse
        elif width > height * 1.5:  # Significantly wider than tall
            # Count cells in top vs bottom row
            top_count = sum(1 for r, c in rel_positions if r == 0)
            bottom_count = sum(1 for r, c in rel_positions if r == height - 1)

            # If bottom has more cells, reverse so it becomes the outer edge
            if bottom_count > top_count:
                rel_positions_adj = set()
                for r, c in rel_positions:
                    rel_positions_adj.add((height - 1 - r, c))
                rel_positions = rel_positions_adj

        # Pad pattern to square (left-aligned)
        square_pattern = [[background for _ in range(max_dim)] for _ in range(max_dim)]
        for r, c in rel_positions:
            square_pattern[r][c] = color

        # Extract positions for symmetry
        sym_positions = set()
        for r in range(max_dim):
            for c in range(max_dim):
                if square_pattern[r][c] == color:
                    sym_positions.add((r, c))

        # Add reflections
        for r, c in list(sym_positions):
            # Vertical reflection about center
            sym_positions.add((max_dim - 1 - r, c))
            # Horizontal reflection about center
            sym_positions.add((r, max_dim - 1 - c))
            # Both reflections
            sym_positions.add((max_dim - 1 - r, max_dim - 1 - c))

        # Create final symmetric pattern
        pattern = [[background for _ in range(max_dim)] for _ in range(max_dim)]
        for r, c in sym_positions:
            pattern[r][c] = color

        patterns.append((max_dim, color, pattern))

    # Sort by max dimension descending
    patterns.sort(key=lambda x: -x[0])

    # Output size is the maximum dimension
    output_size = patterns[0][0]

    # Initialize output with background
    result = [[background for _ in range(output_size)] for _ in range(output_size)]

    # Overlay patterns from largest to smallest, centering each one
    for max_dim, color, pattern in patterns:
        # Calculate offset to center this pattern in the output
        pattern_size = len(pattern)
        offset_r = (output_size - pattern_size) // 2
        offset_c = (output_size - pattern_size) // 2

        # Overlay the pattern (only non-background cells)
        for r in range(pattern_size):
            for c in range(pattern_size):
                if pattern[r][c] != background:
                    result[offset_r + r][offset_c + c] = pattern[r][c]

    return result
