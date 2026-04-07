def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains markers (isolated cells) and pattern definitions (connected groups)
    2. Each marker value maps to a specific output pattern value:
       - Marker 2 → Pattern value 4 (hollow rectangle)
       - Marker 3 → Pattern value 1 (diamond)
       - Marker 5 → Pattern value 6 (2x2 blocks)
       - Marker 8 → Pattern value 7 (X pattern)
    3. Custom patterns in input override default patterns if values match
    4. Patterns are placed with top-left corner at marker position

    Procedure:
    1. Find connected components to identify markers vs patterns
    2. Extract custom pattern structures (if any)
    3. Define default pattern templates for each marker type
    4. For each marker, place the appropriate pattern (custom or default)
    5. Return the output grid
    """
    from collections import deque

    rows, cols = len(grid), len(grid[0])

    # Default 4x4 patterns for each marker type
    default_patterns = {
        2: (4, [(0,0), (0,1), (0,2), (0,3), (1,0), (1,3), (2,0), (2,3), (3,0), (3,1), (3,2), (3,3)]),  # Hollow rectangle
        3: (1, [(0,1), (0,2), (1,0), (1,3), (2,0), (2,3), (3,1), (3,2)]),  # Diamond
        5: (6, [(0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3)]),  # 2x2 blocks diagonal
        8: (7, [(0,0), (0,3), (1,1), (1,2), (2,1), (2,2), (3,0), (3,3)])   # X pattern
    }

    # Find connected components using BFS
    visited = [[False] * cols for _ in range(rows)]
    markers = []
    custom_patterns = {}

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and not visited[r][c]:
                # BFS to find component
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True
                value = grid[r][c]

                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))

                    # Use 8-directional connectivity (including diagonals) for pattern extraction
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] == value:
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                # Single cell = marker, multiple cells = custom pattern
                if len(component) == 1:
                    markers.append((r, c, value))
                else:
                    # Extract pattern structure
                    min_r = min(r for r, c in component)
                    min_c = min(c for r, c in component)
                    # Normalize to relative positions from top-left
                    relative = [(r - min_r, c - min_c) for r, c in component]
                    custom_patterns[value] = relative

    # Create output grid
    output = [[0] * cols for _ in range(rows)]

    # Sort markers by (marker_value, -row, -col) for bottom-right to top-left processing
    # This ensures that markers closer to bottom-right are processed first
    markers.sort(key=lambda x: (x[2], -x[0], -x[1]))

    # Place patterns at marker positions, checking for conflicts
    for mr, mc, marker_value in markers:
        if marker_value in default_patterns:
            pattern_value, pattern_coords = default_patterns[marker_value]

            # Use custom pattern if one exists with the target output value
            if pattern_value in custom_patterns:
                pattern_coords = custom_patterns[pattern_value]

            # Check if pattern area is clear (no conflicts with already-placed patterns)
            can_place = True
            for dr, dc in pattern_coords:
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if output[nr][nc] != 0:
                        can_place = False
                        break

            # Place pattern only if area is clear
            if can_place:
                for dr, dc in pattern_coords:
                    nr, nc = mr + dr, mc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        output[nr][nc] = pattern_value

    return output
