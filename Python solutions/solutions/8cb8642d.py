def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains rectangular regions filled with a uniform color
    2. Each region has exactly one cell with a different "marker" color
    3. Output preserves the boundary of each region but clears the interior
    4. A diamond/X pattern is drawn in the interior using the marker color
    5. On center row(s), the diamond span is filled; on other rows, only endpoints are marked

    Procedure:
    1. Find connected components of non-zero cells (BFS/DFS)
    2. For each component that forms a filled rectangle:
       - Identify the majority color (region/boundary color) and minority color (marker)
       - Clear the interior (keep boundary intact)
       - Draw diamond pattern: for each interior row i, mark at distance d = min(i, height-1-i) from edges
       - Fill the entire span on center row(s); mark only endpoints on other rows
    """

    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    visited = [[False] * w for _ in range(h)]

    def bfs(start_r, start_c):
        """Find all cells in the connected component starting from (start_r, start_c)"""
        queue = [(start_r, start_c)]
        visited[start_r][start_c] = True
        cells = [(start_r, start_c)]

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != 0:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
                    cells.append((nr, nc))

        return cells

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 and not visited[r][c]:
                cells = bfs(r, c)

                # Get bounding box
                min_r = min(cell[0] for cell in cells)
                max_r = max(cell[0] for cell in cells)
                min_c = min(cell[1] for cell in cells)
                max_c = max(cell[1] for cell in cells)

                # Check if it's a filled rectangle
                expected_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
                if len(cells) == expected_cells:
                    # Get colors in this region
                    colors = {}
                    for rr, cc in cells:
                        color = grid[rr][cc]
                        colors[color] = colors.get(color, 0) + 1

                    # Check if there are exactly 2 colors (region color + marker)
                    if len(colors) == 2:
                        # Find majority (region) and minority (marker) colors
                        sorted_colors = sorted(colors.items(), key=lambda x: -x[1])
                        region_color = sorted_colors[0][0]
                        marker_color = sorted_colors[1][0]

                        # Clear interior (preserve boundary)
                        for rr in range(min_r + 1, max_r):
                            for cc in range(min_c + 1, max_c):
                                result[rr][cc] = 0

                        # Draw diamond pattern
                        interior_h = max_r - min_r - 1
                        for i in range(interior_h):
                            # Distance from nearest edge (top or bottom)
                            d = min(i, interior_h - 1 - i)
                            left_col = min_c + 1 + d
                            right_col = max_c - 1 - d

                            # Determine if this is a center row
                            is_center = False
                            if interior_h % 2 == 1:
                                # Odd height: one center row
                                is_center = (i == interior_h // 2)
                            else:
                                # Even height: two center rows
                                is_center = (i == interior_h // 2 - 1 or i == interior_h // 2)

                            if is_center:
                                # Fill the entire span on center row(s)
                                for cc in range(left_col, right_col + 1):
                                    result[min_r + 1 + i][cc] = marker_color
                            else:
                                # Mark only endpoints on non-center rows
                                result[min_r + 1 + i][left_col] = marker_color
                                if left_col != right_col:
                                    result[min_r + 1 + i][right_col] = marker_color

    return result
