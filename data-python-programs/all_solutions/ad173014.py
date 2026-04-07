def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with multiple rectangular frames bounded by 2s
    2. Each frame contains a colored pattern inside (colors: 3, 4, 6, 7, 8, etc.)
    3. Output shows the same frames but with colors rotated between frames
    4. Each frame keeps its pattern structure but gets the color from the next frame in clockwise order
    5. Clockwise order starts from the topmost frame and proceeds spatially clockwise

    Procedure:
    1. Find all rectangular frames (regions bounded by 2s)
    2. Extract the internal patterns from each frame
    3. Sort frames by spatial position (clockwise from top-left)
    4. Rotate contents: each frame gets pattern from next frame clockwise
    """

    def find_frames(grid):
        """Find all rectangular frames bounded by 2s"""
        rows, cols = len(grid), len(grid[0])
        visited = [[False] * cols for _ in range(rows)]
        frames = []

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2 and not visited[r][c]:
                    frame = trace_frame(grid, r, c, visited)
                    if frame:
                        frames.append(frame)

        return frames

    def trace_frame(grid, start_r, start_c, visited):
        """Trace a rectangular frame starting from a 2"""
        rows, cols = len(grid), len(grid[0])

        # Find the bounding box of connected 2s
        queue = [(start_r, start_c)]
        visited[start_r][start_c] = True
        frame_cells = [(start_r, start_c)]

        while queue:
            r, c = queue.pop(0)
            # Check 4 directions
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr][nc]
                    and grid[nr][nc] == 2
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    frame_cells.append((nr, nc))

        # Find bounding box
        min_r = min(r for r, c in frame_cells)
        max_r = max(r for r, c in frame_cells)
        min_c = min(c for r, c in frame_cells)
        max_c = max(c for r, c in frame_cells)

        # Verify it's a proper rectangular frame
        if is_rectangular_frame(grid, min_r, max_r, min_c, max_c):
            return (min_r, max_r, min_c, max_c)

        return None

    def is_rectangular_frame(grid, min_r, max_r, min_c, max_c):
        """Check if the region forms a proper rectangular frame"""
        # Check if perimeter is all 2s and interior is not all 2s
        has_interior = False
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if r == min_r or r == max_r or c == min_c or c == max_c:
                    # Perimeter should be 2
                    if grid[r][c] != 2:
                        return False
                else:
                    # Interior - should have some non-2 values
                    if grid[r][c] != 2:
                        has_interior = True

        return has_interior and (max_r - min_r >= 2) and (max_c - min_c >= 2)

    def get_center(frame):
        """Get center coordinates of a frame"""
        min_r, max_r, min_c, max_c = frame
        return ((min_r + max_r) / 2, (min_c + max_c) / 2)

    def sort_four_frames_clockwise(frames):
        """Sort 4 frames in clockwise order starting from top, then going clockwise"""
        import math

        centers = [get_center(frame) for frame in frames]

        # Find the topmost frame first
        top_frame_idx = min(range(len(centers)), key=lambda i: centers[i][0])
        top_frame = frames[top_frame_idx]
        top_center = centers[top_frame_idx]

        # Remove the top frame from consideration
        remaining_frames = [frames[i] for i in range(len(frames)) if i != top_frame_idx]
        remaining_centers = [centers[i] for i in range(len(centers)) if i != top_frame_idx]

        # Sort remaining frames by angle from top frame center
        def angle_from_top(center):
            dy = center[0] - top_center[0]
            dx = center[1] - top_center[1]
            # Calculate angle from top frame, going clockwise
            angle = math.atan2(dy, dx)
            # Adjust so that right is 0, down is pi/2, left is pi, up is 3*pi/2
            angle = (angle + math.pi / 2) % (2 * math.pi)
            return angle

        # Sort by angle clockwise from the top
        indexed_remaining = list(zip(remaining_frames, remaining_centers))
        indexed_remaining.sort(key=lambda x: angle_from_top(x[1]))

        # Construct final order: top frame first, then others in clockwise order
        result = [top_frame]
        result.extend([frame for frame, center in indexed_remaining])

        return result

    def sort_frames_clockwise(frames):
        """Sort frames in clockwise order starting from top-left"""

        # Sort frames by their center positions
        # First by row (top first), then by column (left first)
        def sort_key(frame):
            center_r, center_c = get_center(frame)
            return (center_r, center_c)

        # For clockwise ordering, we need to consider the spatial arrangement
        if len(frames) == 3:
            # For 3 frames, sort by position: top, then left-to-right for remaining
            return sorted(frames, key=lambda f: (get_center(f)[0], get_center(f)[1]))
        elif len(frames) == 4:
            # For 4 frames, find corners and sort clockwise
            return sort_four_frames_clockwise(frames)
        else:
            # Default sorting by position
            return sorted(frames, key=sort_key)

    def extract_pattern(grid, frame):
        """Extract the internal pattern from a frame"""
        min_r, max_r, min_c, max_c = frame
        pattern = []

        for r in range(min_r + 1, max_r):
            row = []
            for c in range(min_c + 1, max_c):
                row.append(grid[r][c])
            pattern.append(row)

        return pattern

    def extract_main_color(pattern):
        """Extract the main color (non-zero) from a pattern"""
        for row in pattern:
            for cell in row:
                if cell != 0:
                    return cell
        return 0

    def apply_pattern_with_color(grid, frame, pattern, color):
        """Apply a pattern to the interior of a frame, replacing non-zero values with the new color"""
        min_r, max_r, min_c, max_c = frame

        for i, r in enumerate(range(min_r + 1, max_r)):
            for j, c in enumerate(range(min_c + 1, max_c)):
                if i < len(pattern) and j < len(pattern[i]):
                    if pattern[i][j] == 0:
                        grid[r][c] = 0
                    else:
                        grid[r][c] = color

    # Create a copy of the grid to modify
    result = [row[:] for row in grid]

    # Find all frames bounded by 2s
    frames = find_frames(grid)

    # Sort frames by spatial position (clockwise)
    frames = sort_frames_clockwise(frames)

    # Extract patterns and colors from each frame
    patterns = []
    colors = []
    for frame in frames:
        pattern = extract_pattern(grid, frame)
        color = extract_main_color(pattern)
        patterns.append(pattern)
        colors.append(color)

    # Each frame keeps its pattern structure but gets color from next frame
    for i, frame in enumerate(frames):
        next_color = colors[(i + 1) % len(colors)]
        apply_pattern_with_color(result, frame, patterns[i], next_color)

    return result
