def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains line segments (horizontal, vertical, or diagonal down-left)
    2. All segments are normalized to the second-longest unique segment length
    3. Example 1: diagonal segments with lengths [3,2,6,9,4] → normalized to 6 (second-longest)
    4. Example 2: horizontal segments with lengths [2,7,12] → normalized to 7
    5. Example 3: vertical segments with lengths [2,4,3,2,6] → normalized to 4

    Procedure:
    1. Find all line segments in each direction (horizontal, vertical, diagonal)
    2. Calculate second-longest unique length across all segments
    3. Normalize each segment to the target length (extend or truncate from start point)
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find all segments in all three directions
    segments = []
    visited = set()

    directions = [
        (1, -1),  # diagonal (down-left) - check first
        (0, 1),   # horizontal (right)
        (1, 0),   # vertical (down)
    ]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 7 and (r, c) not in visited:
                color = grid[r][c]

                # Try each direction to find a segment starting from this cell
                for dr, dc in directions:
                    # Check if this is the start of a segment in this direction
                    # (no cell before it in opposite direction with same color)
                    prev_r, prev_c = r - dr, c - dc
                    if 0 <= prev_r < rows and 0 <= prev_c < cols and grid[prev_r][prev_c] == color:
                        continue  # Not the start in this direction

                    # Trace the segment forward
                    segment = []
                    cr, cc = r, c
                    while 0 <= cr < rows and 0 <= cc < cols and grid[cr][cc] == color:
                        segment.append((cr, cc))
                        visited.add((cr, cc))
                        cr += dr
                        cc += dc

                    if len(segment) >= 2:
                        segments.append((color, segment, dr, dc))
                        break  # Found a segment, no need to check other directions

    # Find second-longest unique length
    lengths = sorted(set(len(seg) for _, seg, _, _ in segments), reverse=True)
    target_length = lengths[1] if len(lengths) >= 2 else (lengths[0] if lengths else 0)

    # Create output grid (start with background)
    result = [[7] * cols for _ in range(rows)]

    # Normalize each segment to target length
    for color, segment, dr, dc in segments:
        start_r, start_c = segment[0]

        if dr == 1 and dc == -1:  # Diagonal segments: prepend (extend backwards)
            # Calculate new start position by going backwards from original start
            new_start_r = start_r - (target_length - len(segment)) * dr
            new_start_c = start_c - (target_length - len(segment)) * dc
            # Place cells from new start position
            for i in range(target_length):
                r = new_start_r + i * dr
                c = new_start_c + i * dc
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = color
        else:  # Horizontal and vertical segments: append (extend forwards)
            # Place cells from original start position
            for i in range(target_length):
                r = start_r + i * dr
                c = start_c + i * dc
                if 0 <= r < rows and 0 <= c < cols:
                    result[r][c] = color

    return result
