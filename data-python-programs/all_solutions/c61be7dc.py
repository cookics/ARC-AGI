def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Each input has lines of all-0s that divide the grid into regions
    2. There's always a central region containing 5s
    3. The transformation "collapses" this central region along one axis
    4. The collapse direction depends on frame structure:
       - If there are 2+ vertical 0-lines (forming left/right frames): collapse horizontally
       - If there are 2+ horizontal 0-lines (forming top/bottom frames): collapse vertically
    5. The orthogonal 0-lines (perpendicular to collapse) remain unchanged
    6. The 5s form a line through the collapsed region with specific extent rules

    Procedure:
    1. Find all frame lines (full rows/columns of 0s)
    2. Determine collapse direction based on frame count
    3. Calculate center position and flanking 0-lines
    4. Place 5s with proper extent (avoid 0-boundaries, extend toward edges)
    5. Restore orthogonal frame lines
    """

    rows, cols = len(grid), len(grid[0])
    result = [[7 for _ in range(cols)] for _ in range(rows)]

    # Find all 5s positions
    fives_positions = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                fives_positions.append((r, c))

    if not fives_positions:
        # No content, just copy 0s
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    result[r][c] = 0
        return result

    # Find vertical frame boundaries (columns of all 0s)
    vertical_frames = []
    for c in range(cols):
        if all(grid[r][c] == 0 for r in range(rows)):
            vertical_frames.append(c)

    # Find horizontal frame boundaries (rows of all 0s)
    horizontal_frames = []
    for r in range(rows):
        if all(grid[r][c] == 0 for c in range(cols)):
            horizontal_frames.append(r)

    # Also find "structural" horizontal frames (rows that are 0s except in content region)
    if len(vertical_frames) >= 2:
        left_bound = vertical_frames[0]
        right_bound = vertical_frames[-1]
        for r in range(rows):
            if r in horizontal_frames:
                continue  # Already found as full 0-line
            is_structural = True
            for c in range(cols):
                if left_bound < c < right_bound:
                    # Inside content region, can be anything
                    continue
                elif grid[r][c] != 0:
                    # Outside content region, must be 0
                    is_structural = False
                    break
            if is_structural:
                horizontal_frames.append(r)

    # Similarly for structural vertical frames
    elif len(horizontal_frames) >= 2:
        top_bound = horizontal_frames[0]
        bottom_bound = horizontal_frames[-1]
        for c in range(cols):
            if c in vertical_frames:
                continue  # Already found as full 0-line
            is_structural = True
            for r in range(rows):
                if top_bound < r < bottom_bound:
                    # Inside content region, can be anything
                    continue
                elif grid[r][c] != 0:
                    # Outside content region, must be 0
                    is_structural = False
                    break
            if is_structural:
                vertical_frames.append(c)

    # Get content bounds
    min_r = min(pos[0] for pos in fives_positions)
    max_r = max(pos[0] for pos in fives_positions)
    min_c = min(pos[1] for pos in fives_positions)
    max_c = max(pos[1] for pos in fives_positions)

    # Determine collapse direction based on frame structure
    if len(vertical_frames) >= 2:
        # Collapse horizontally (squeeze width, move vertical frames inward)
        vertical_frames.sort()
        left_frame = vertical_frames[0]
        right_frame = vertical_frames[-1]

        # Find center column between the frames
        center_col = (left_frame + right_frame) // 2

        # Place flanking 0 columns
        for r in range(rows):
            result[r][center_col - 1] = 0
            result[r][center_col + 1] = 0

        # Determine vertical extent for 5s line
        # Rule: extend from edge toward center, but stop before hitting 0-boundaries
        vertical_start = 0
        vertical_end = rows - 1

        # Stop before any horizontal 0-line boundaries
        for frame_r in horizontal_frames:
            if frame_r <= min_r:
                vertical_start = max(vertical_start, frame_r + 1)
            elif frame_r >= max_r:
                vertical_end = min(vertical_end, frame_r - 1)

        # Refined logic: stop before edges only when content is small and centered
        content_height = max_r - min_r + 1
        if content_height <= 3:  # Small content like example 1 (3x3 block)
            if not any(frame_r <= min_r for frame_r in horizontal_frames):
                vertical_start = 1
            if not any(frame_r >= max_r for frame_r in horizontal_frames):
                vertical_end = rows - 2

        # Place 5s in center column
        for r in range(vertical_start, vertical_end + 1):
            result[r][center_col] = 5

        # Restore horizontal frame lines (but preserve 5s at intersections)
        for r in horizontal_frames:
            for c in range(cols):
                if c != center_col:  # Don't overwrite the 5s line
                    result[r][c] = 0
                else:
                    # Keep the 5 if it's within the 5s range, otherwise make it 0
                    if vertical_start <= r <= vertical_end:
                        result[r][c] = 5
                    else:
                        result[r][c] = 0

    elif len(horizontal_frames) >= 2:
        # Collapse vertically (squeeze height, move horizontal frames inward)
        horizontal_frames.sort()
        top_frame = horizontal_frames[0]
        bottom_frame = horizontal_frames[-1]

        # Find center row between the frames
        center_row = (top_frame + bottom_frame) // 2

        # Place flanking 0 rows
        for c in range(cols):
            result[center_row - 1][c] = 0
            result[center_row + 1][c] = 0

        # Determine horizontal extent for 5s line
        # Rule: extend from content bounds with some expansion
        horizontal_start = max(0, min_c - 1)
        horizontal_end = min(cols - 1, max_c + 1)

        # Stop before any vertical 0-line boundaries
        for frame_c in vertical_frames:
            if frame_c <= min_c:
                horizontal_start = max(horizontal_start, frame_c + 1)
            elif frame_c >= max_c:
                horizontal_end = min(horizontal_end, frame_c - 1)

        # Place 5s in center row
        for c in range(horizontal_start, horizontal_end + 1):
            result[center_row][c] = 5

        # Restore vertical frame lines (but preserve 5s at intersections)
        for c in vertical_frames:
            for r in range(rows):
                if r != center_row:  # Don't overwrite the 5s line
                    result[r][c] = 0
                else:
                    # Keep the 5 if it's within the 5s range, otherwise make it 0
                    if horizontal_start <= c <= horizontal_end:
                        result[r][c] = 5
                    else:
                        result[r][c] = 0

    return result
