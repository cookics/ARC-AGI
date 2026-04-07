def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains multiple separate patterns made of 1s on a background of 4s
    2. Each pattern needs to be moved to either the left edge or right edge
    3. Patterns moved left have their 1s changed to 2s
    4. Patterns moved right have their 1s changed to 3s
    5. The internal structure of each pattern is preserved (including 4s inside)
    6. Assignment to left/right follows a specific pattern based on total count:
       - Even N: first N/2 left, last N/2 right
       - Odd N where N%4==1: first 2 left, then alternate R,L,R,L,...
       - Odd N where N%4==3: first 2 right, then alternate L,R,L,R,...

    Procedure:
    1. Find all connected components/patterns of 1s
    2. Extract bounding box and content for each pattern
    3. Sort patterns by their minimum row
    4. Assign each pattern to LEFT (color 2) or RIGHT (color 3)
    5. Place patterns at the appropriate edge in the output grid
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Copy input

    # Find all patterns (regions containing 1s)
    visited = [[False] * cols for _ in range(rows)]
    patterns = []

    def get_bounding_box(r, c):
        """Get bounding box of a pattern starting from (r, c)"""
        cells = []
        stack = [(r, c)]
        visited[r][c] = True

        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))

            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if (0 <= nr < rows and 0 <= nc < cols and
                    not visited[nr][nc] and grid[nr][nc] == 1):
                    visited[nr][nc] = True
                    stack.append((nr, nc))

        if not cells:
            return None

        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Extract pattern content
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        pattern_content = [[grid[min_r + i][min_c + j] for j in range(width)]
                          for i in range(height)]

        return {
            'min_r': min_r,
            'max_r': max_r,
            'min_c': min_c,
            'max_c': max_c,
            'content': pattern_content,
            'height': height,
            'width': width
        }

    # Find all patterns
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                pattern = get_bounding_box(r, c)
                if pattern:
                    patterns.append(pattern)

    # Sort patterns by minimum row
    patterns.sort(key=lambda p: p['min_r'])

    N = len(patterns)

    # Determine assignment for each pattern
    assignments = []  # True for LEFT, False for RIGHT

    if N % 2 == 0:
        # Even: first half left, second half right
        for i in range(N):
            assignments.append(i < N // 2)
    else:
        # Odd
        if N % 4 == 1:
            # Pattern: L, L, R, L, R, L, R, ...
            for i in range(N):
                if i < 2:
                    assignments.append(True)  # LEFT
                else:
                    # Alternate starting with RIGHT
                    assignments.append((i - 2) % 2 == 1)  # R, L, R, L, ...
        else:  # N % 4 == 3
            # Pattern: R, R, L, R, L, R, L, ...
            for i in range(N):
                if i < 2:
                    assignments.append(False)  # RIGHT
                else:
                    # Alternate starting with LEFT
                    assignments.append((i - 2) % 2 == 0)  # L, R, L, R, ...

    # Clear all 1s from result
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == 1:
                result[r][c] = 4

    # Place each pattern at the appropriate edge
    for i, pattern in enumerate(patterns):
        go_left = assignments[i]
        color = 2 if go_left else 3

        height = pattern['height']
        width = pattern['width']
        content = pattern['content']
        row_start = pattern['min_r']

        if go_left:
            # Place at left edge (column 0)
            col_start = 0
        else:
            # Place at right edge
            col_start = cols - width

        # Copy pattern to result
        for dr in range(height):
            for dc in range(width):
                if content[dr][dc] == 1:
                    result[row_start + dr][col_start + dc] = color
                # Note: 4s inside the pattern remain as 4s in the result

    return result
