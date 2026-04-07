def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains hollow rectangular regions (large connected components with interior holes)
    2. Input contains small external patterns (small connected components outside rectangles)
    3. Value 5 appears as markers/noise scattered throughout the grid
    4. Output removes all 5s and clears external patterns from their original positions
    5. Output places external patterns into the interior holes of rectangular regions

    Procedure:
    1. Remove all cells with value 5 from the grid
    2. Find all connected components using DFS
    3. Classify components as rectangular regions (large, hollow) or external patterns (small)
    4. Clear external patterns from their original positions
    5. Place each external pattern into the interior space of a rectangular region
    """

    rows, cols = len(grid), len(grid[0])

    # Step 1: Remove all 5s
    result = [
        [0 if grid[i][j] == 5 else grid[i][j] for j in range(cols)] for i in range(rows)
    ]

    # Step 2: Find all connected components
    visited = [[False] * cols for _ in range(rows)]
    components = []

    def dfs(r, c, value, component):
        if (
            r < 0
            or r >= rows
            or c < 0
            or c >= cols
            or visited[r][c]
            or result[r][c] != value
        ):
            return
        visited[r][c] = True
        component.append((r, c))
        dfs(r + 1, c, value, component)
        dfs(r - 1, c, value, component)
        dfs(r, c + 1, value, component)
        dfs(r, c - 1, value, component)

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and result[i][j] != 0:
                component = []
                dfs(i, j, result[i][j], component)
                if component:
                    components.append(
                        {
                            "cells": set(component),
                            "value": result[i][j],
                            "size": len(component),
                        }
                    )

    # Step 3: Classify as rectangular regions vs external patterns
    components.sort(key=lambda x: x["size"], reverse=True)

    # Find rectangular regions (large components that form rectangles)
    rectangular_regions = []
    external_patterns = []

    for comp in components:
        cells = list(comp["cells"])
        if len(cells) < 8:  # Too small to be a rectangular region
            external_patterns.append(comp)
            continue

        # Check if it forms a rectangular boundary
        min_r = min(r for r, c in cells)
        max_r = max(r for r, c in cells)
        min_c = min(c for r, c in cells)
        max_c = max(c for r, c in cells)

        # Check if it's a rectangular frame (has interior space)
        is_rectangular_region = False
        interior_cells = []

        for r in range(min_r + 1, max_r):
            for c in range(min_c + 1, max_c):
                if (r, c) not in comp["cells"]:
                    interior_cells.append((r, c))
                    is_rectangular_region = True

        if is_rectangular_region and len(cells) > 15:
            rectangular_regions.append(
                {
                    "cells": comp["cells"],
                    "value": comp["value"],
                    "interior": interior_cells,
                    "bounds": (min_r, max_r, min_c, max_c),
                }
            )
        else:
            external_patterns.append(comp)

    # Step 4: Clear external patterns from result
    for pattern in external_patterns:
        for r, c in pattern["cells"]:
            result[r][c] = 0

    # Step 5: Place external patterns into rectangular regions
    for pattern in external_patterns:
        pattern_cells = list(pattern["cells"])
        pattern_value = pattern["value"]

        # Get pattern shape
        if not pattern_cells:
            continue

        min_pr = min(r for r, c in pattern_cells)
        max(r for r, c in pattern_cells)
        min_pc = min(c for r, c in pattern_cells)
        max(c for r, c in pattern_cells)

        # Try to place in each rectangular region
        placed = False
        for region in rectangular_regions:
            if placed:
                break

            # Try different positions within the region's interior
            for target_r in range(region["bounds"][0] + 1, region["bounds"][1]):
                for target_c in range(region["bounds"][2] + 1, region["bounds"][3]):
                    # Check if pattern fits at this position
                    can_place = True
                    positions_to_fill = []

                    for pr, pc in pattern_cells:
                        new_r = target_r + (pr - min_pr)
                        new_c = target_c + (pc - min_pc)

                        if (
                            new_r < region["bounds"][0] + 1
                            or new_r >= region["bounds"][1]
                            or new_c < region["bounds"][2] + 1
                            or new_c >= region["bounds"][3]
                            or (new_r, new_c) in region["cells"]
                            or result[new_r][new_c] != 0
                        ):
                            can_place = False
                            break
                        positions_to_fill.append((new_r, new_c))

                    if can_place:
                        # Place the pattern
                        for nr, nc in positions_to_fill:
                            result[nr][nc] = pattern_value
                        placed = True
                        break
                if placed:
                    break

    return result
