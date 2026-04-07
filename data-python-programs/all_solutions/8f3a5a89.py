def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 8s (background), 1s (barriers), and one 6 (starting marker)
    2. Flood fill from 6 through 8s (1s act as barriers)
    3. Visited cells adjacent to non-visited cells or boundary become 7
    4. Visited interior cells become 8, the 6 stays as 6
    5. Only 1s adjacent to visited region are preserved, others become 8

    Procedure:
    1. Find position of 6
    2. BFS flood fill from 6, visiting only 8s (1s are barriers)
    3. Mark visited cells as 7 if on boundary, else 8
    4. Preserve only 1s adjacent to visited cells
    """
    from collections import deque

    rows = len(grid)
    cols = len(grid[0])

    # Find the position of 6
    start_r, start_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 6:
                start_r, start_c = r, c
                break
        if start_r is not None:
            break

    # BFS flood fill from 6
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(start_r, start_c)])
    visited[start_r][start_c] = True

    while queue:
        r, c = queue.popleft()

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                if grid[nr][nc] == 8:  # Only visit 8s (1s are barriers)
                    visited[nr][nc] = True
                    queue.append((nr, nc))

    # Create output grid - initialize to all 8s
    result = [[8] * cols for _ in range(rows)]

    # Process visited cells
    for r in range(rows):
        for c in range(cols):
            if visited[r][c]:
                if grid[r][c] == 6:
                    result[r][c] = 6  # Preserve the marker
                else:
                    # Check if adjacent to non-visited cell or grid boundary
                    # Use 8-connectivity (including diagonals) for boundary detection
                    is_boundary = False

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nr, nc = r + dr, c + dc
                        # Boundary if adjacent to grid edge or non-visited cell
                        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not visited[nr][nc]:
                            is_boundary = True
                            break

                    result[r][c] = 7 if is_boundary else 8

    # Process 1s - only preserve those adjacent to visited cells
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                # Check if adjacent to visited cell
                adjacent_to_visited = False
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and visited[nr][nc]:
                        adjacent_to_visited = True
                        break

                if adjacent_to_visited:
                    result[r][c] = 1
                # else: already set to 8

    return result
