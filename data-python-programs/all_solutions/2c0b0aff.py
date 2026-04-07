def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with multiple rectangular regions containing 8s and 3s separated by 0s
    2. Output is one selected region based on dimensional criteria
    3. When multiple square regions (8x8) exist, the second one is selected
    4. Otherwise rectangular regions (width != height) are preferred over square ones
    5. If no height-8 regions exist, height-7 regions are used as fallback

    Procedure:
    1. Find all rectangular regions using connected component analysis
    2. Filter regions by height, preferring height 8 or falling back to height 7
    3. Classify regions as rectangular or square based on dimensions
    4. Select second square region if multiple exist, otherwise first rectangular region
    5. Return the selected region grid
    """

    # Find all rectangular regions
    regions = []
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] != 0 and not visited[i][j]:
                # BFS to find bounding box
                min_row = max_row = i
                min_col = max_col = j
                queue = [(i, j)]

                while queue:
                    r, c = queue.pop(0)
                    if visited[r][c]:
                        continue
                    visited[r][c] = True

                    min_row = min(min_row, r)
                    max_row = max(max_row, r)
                    min_col = min(min_col, c)
                    max_col = max(max_col, c)

                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 0 and not visited[nr][nc]:
                            queue.append((nr, nc))

                # Extract region
                region = [[grid[r][c] for c in range(min_col, max_col + 1)] for r in range(min_row, max_row + 1)]
                regions.append({
                    "region": region,
                    "size": (max_row - min_row + 1, max_col - min_col + 1)
                })

    # Filter by height
    height8_regions = [r for r in regions if r["size"][0] == 8]

    if not height8_regions:
        height7_regions = [r for r in regions if r["size"][0] == 7]
        if height7_regions:
            result = max(height7_regions, key=lambda r: r["size"][1])["region"]
            return result

    # Classify regions
    rectangular_regions = [r for r in height8_regions if r["size"][0] != r["size"][1]]
    square_regions = [r for r in height8_regions if r["size"][0] == r["size"][1]]

    # Apply selection rules
    if len(square_regions) >= 2:
        result = square_regions[1]["region"]
    elif rectangular_regions:
        result = rectangular_regions[0]["region"]
    elif len(square_regions) >= 1:
        result = square_regions[0]["region"]
    elif regions:
        result = max(regions, key=lambda r: r["size"][0] * r["size"][1])["region"]
    else:
        result = [[]]

    return result
