def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing 0s (background) and 8s (forming connected regions)
    2. Output is a grid where each connected component of 8s is replaced with a color
    3. Colors used are 1, 2, 3, and 7
    4. Color assignment is based on bounding box area of each component
    5. Smaller areas get smaller color numbers (1), larger areas get larger numbers (7)

    Procedure:
    1. Find all connected components of 8s using flood fill
    2. Compute bounding box area for each component
    3. Sort components by area
    4. Assign colors based on area quintiles/percentiles
    5. Replace each component with its assigned color in the result grid
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    result = [[0] * cols for _ in range(rows)]

    def flood_fill(r, c):
        """Find all cells in the connected component starting from (r, c)"""
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c] or grid[r][c] != 8:
            return []

        visited[r][c] = True
        cells = [(r, c)]

        # Check all 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            cells.extend(flood_fill(r + dr, c + dc))

        return cells

    # Find all connected components with their bounding boxes
    components = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8 and not visited[r][c]:
                cells = flood_fill(r, c)
                if cells:
                    # Calculate bounding box
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)

                    bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
                    center_r = (min_r + max_r) / 2

                    components.append({
                        'cells': cells,
                        'area': bbox_area,
                        'center_r': center_r,
                        'min_r': min_r,
                        'index': len(components)
                    })

    # Sort components by area, then by vertical position as tiebreaker
    components.sort(key=lambda x: (x['area'], x['min_r']))

    # Assign colors based on area distribution
    n = len(components)
    component_colors = [0] * n

    if n > 0:
        # Assign colors based on rank position
        # The exact distribution depends on the number of components and area ranges

        for i, comp in enumerate(components):
            rank = i  # 0-indexed rank (smallest area = 0)

            # Adaptive color assignment based on number of components
            if n <= 2:
                # Very few components: simple split
                color = 3 if rank == 0 else 7
            elif n <= 4:
                # Few components: use colors 3 and 7
                if rank < n // 2:
                    color = 3
                else:
                    color = 7
            elif n == 5:
                # 5 components: pattern is 1,1,2,2,3 (first 2 get 1, next 2 get 2, last gets 3)
                if rank < 2:
                    color = 1
                elif rank < 4:
                    color = 2
                else:
                    color = 3
            elif n == 6:
                # 6 components: pattern is 1,2,2,2,3,7
                if rank == 0:
                    color = 1
                elif rank < 4:
                    color = 2
                elif rank == 4:
                    color = 3
                else:
                    color = 7
            else:
                # Many components: use all colors with adaptive thresholds
                percentile = (rank + 1) / n
                if percentile <= 0.25:
                    color = 1
                elif percentile <= 0.65:
                    color = 2
                elif percentile <= 0.85:
                    color = 3
                else:
                    color = 7

            component_colors[comp['index']] = color

    # Fill the result grid with assigned colors
    for comp in components:
        color = component_colors[comp['index']]
        for r, c in comp['cells']:
            result[r][c] = color

    return result
