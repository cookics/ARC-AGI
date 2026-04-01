def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Looking at the expected patterns, each value seems to create specific structures:
    - Value 2: appears to form inverted triangular or arrow patterns pointing up-left
    - Value 5: appears to form triangular patterns pointing down-right
    - Value 8: stays in original positions
    - Value 9: behaves like value 5

    Procedure:
    Apply pattern-based transformation for each value type based on their positions.
    """

    h, w = len(grid), len(grid[0])
    result = [[7 for _ in range(w)] for _ in range(h)]

    # Process 8s first (they stay in place)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 8:
                result[r][c] = 8

    # Process 2s - they seem to create upward-left pointing patterns
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                # Create a pattern that extends up-left but with limited extent
                max_dist = min(3, r, c)  # Limit to 3 steps or boundary
                for d in range(max_dist + 1):
                    nr, nc = r - d, c - d
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == 7:
                        result[nr][nc] = 2

    # Process 5s and 9s - they seem to create downward-right pointing patterns
    for r in range(h):
        for c in range(w):
            if grid[r][c] in [5, 9]:
                value = grid[r][c]
                # Create a pattern that extends down-right but with limited extent
                max_dist = min(3, h - r - 1, w - c - 1)  # Limit to 3 steps or boundary
                for d in range(max_dist + 1):
                    nr, nc = r + d, c + d
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == 7:
                        result[nr][nc] = value

    return result


def find_connected_components(grid, value):
    h, w = len(grid), len(grid[0])
    visited = set()
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r][c] == value and (r, c) not in visited:
                component = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (
                        (cr, cc) in visited
                        or cr < 0
                        or cr >= h
                        or cc < 0
                        or cc >= w
                        or grid[cr][cc] != value
                    ):
                        continue
                    visited.add((cr, cc))
                    component.append((cr, cc))
                    # Use 8-connectivity
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr != 0 or dc != 0:
                                stack.append((cr + dr, cc + dc))
                components.append(component)
    return components


def expand_diagonal(component):
    """Expand a component diagonally"""
    result = list(component)

    # Calculate bounding box
    min_r = min(r for r, c in component)
    max_r = max(r for r, c in component)
    min_c = min(c for r, c in component)
    max_c = max(c for r, c in component)

    # Expand in all four diagonal directions
    for r, c in component:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_r, new_c = r + dr, c + dc
            result.append((new_r, new_c))

    return list(set(result))
