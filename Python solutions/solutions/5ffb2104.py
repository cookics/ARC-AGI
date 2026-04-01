def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with non-zero values forming connected components scattered across the grid.
    2. Output is a grid where all non-zero connected components are moved to the rightmost positions.
    3. Each component maintains its original shape and vertical position.
    4. Components are placed as far right as possible without overlapping each other.
    5. Components originally on the left are placed first (rightmost), then leftward components follow.

    Procedure:
    1. Find all connected components using flood fill (4-connected).
    2. Sort components by their leftmost column position.
    3. Process components in reverse order (rightmost first).
    4. For each component, find the rightmost position where it can fit without collision.
    5. Place the component at that position.
    """

    height = len(grid)
    width = len(grid[0])

    # Find connected components using flood fill
    visited = [[False] * width for _ in range(height)]
    components = []

    def flood_fill(start_r, start_c):
        component = []
        stack = [(start_r, start_c)]

        while stack:
            r, c = stack.pop()
            if (
                r < 0
                or r >= height
                or c < 0
                or c >= width
                or visited[r][c]
                or grid[r][c] == 0
            ):
                continue

            visited[r][c] = True
            component.append((r, c, grid[r][c]))

            # Check 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))

        return component

    # Find all connected components
    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0 and not visited[r][c]:
                comp = flood_fill(r, c)
                if comp:
                    components.append(comp)

    if not components:
        return [[0] * width for _ in range(height)]

    # Sort components by leftmost column position
    components.sort(key=lambda comp: min(c for r, c, v in comp))

    # Create output grid
    result = [[0] * width for _ in range(height)]

    # Place components from rightmost to leftmost to avoid conflicts
    for comp in reversed(components):
        # Find bounding box of component
        min_r = min(r for r, c, v in comp)
        max_r = max(r for r, c, v in comp)
        min_c = min(c for r, c, v in comp)
        max_c = max(c for r, c, v in comp)

        comp_width = max_c - min_c + 1

        # Find rightmost position where component fits without collision
        best_target_c = None
        for target_c in range(width - comp_width, -1, -1):
            # Check if placing component at target_c causes collision
            collision = False
            for r, c, v in comp:
                new_r = r
                new_c = c - min_c + target_c
                if result[new_r][new_c] != 0:
                    collision = True
                    break

            if not collision:
                best_target_c = target_c
                break

        # Place the component in the found position
        if best_target_c is not None:
            for r, c, v in comp:
                new_r = r
                new_c = c - min_c + best_target_c
                result[new_r][new_c] = v

    return result
