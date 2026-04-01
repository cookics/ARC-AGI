def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid containing colored rectangular blocks at different vertical positions
    2. Output has all blocks aligned to the same vertical position as a designated reference block
    3. Each block maintains its original horizontal position (columns)
    4. Blocks are connected components of the same non-zero color

    Procedure:
    1. Find all rectangular blocks by identifying connected components of same color using DFS
    2. Determine the bounding box (min/max row and column) of each block
    3. Identify the target vertical position from the designated reference block
    4. Create output grid initialized with zeros
    5. Reposition all blocks to align with the target vertical position while preserving horizontal positions
    """

    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    blocks = []

    # Find all connected components (blocks)
    def dfs(r, c, color, cells):
        if (
            r < 0
            or r >= rows
            or c < 0
            or c >= cols
            or visited[r][c]
            or grid[r][c] != color
        ):
            return
        visited[r][c] = True
        cells.append((r, c))
        dfs(r + 1, c, color, cells)
        dfs(r - 1, c, color, cells)
        dfs(r, c + 1, color, cells)
        dfs(r, c - 1, color, cells)

    # Find all blocks
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                cells = []
                dfs(r, c, grid[r][c], cells)
                if cells:
                    # Get bounding box
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)
                    blocks.append(
                        {
                            "color": grid[r][c],
                            "cells": cells,
                            "min_r": min_r,
                            "max_r": max_r,
                            "min_c": min_c,
                            "max_c": max_c,
                        }
                    )

    # Find the target row range (where 1's block is located)
    target_min_r = None
    target_max_r = None
    for block in blocks:
        if block["color"] == 1:
            target_min_r = block["min_r"]
            target_max_r = block["max_r"]
            break

    # Create output grid
    result = [[0] * cols for _ in range(rows)]

    # Place all blocks at target row range
    for block in blocks:
        color = block["color"]
        # Calculate vertical offset to align with target
        offset = target_min_r - block["min_r"]

        # Place the block at new position
        for r, c in block["cells"]:
            new_r = r + offset
            if 0 <= new_r < rows:
                result[new_r][c] = color

    return result
