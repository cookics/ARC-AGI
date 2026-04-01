def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a checkerboard pattern (0s and 1s) with special values overlaid
    2. Output extends diagonal segments in PERPENDICULAR diagonal directions from endpoints
    3. Value 3 acts as an anchor/center and doesn't propagate
    4. For SE/NW segments, extend in NE/SW perpendicular directions
    5. For SW/NE segments, extend in SE/NW perpendicular directions

    Procedure:
    1. Find all diagonal segments of each special value (not 0, 1, 3)
    2. For each segment with 2+ cells, extend from endpoints in perpendicular directions
    3. Choose perpendicular direction that doesn't immediately hit another special value
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find all special values (excluding 0, 1, 3)
    special_values = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] not in [0, 1, 3]:
                special_values.add(grid[r][c])

    # For each special value, find diagonal segments and extend
    for val in special_values:
        # Try both main diagonal directions (SE and SW)
        for dr, dc in [(1, 1), (1, -1)]:
            visited = set()

            for r in range(rows):
                for c in range(cols):
                    if grid[r][c] == val and (r, c) not in visited:
                        # Build a segment starting from (r, c) in direction (dr, dc)
                        segment = []
                        nr, nc = r, c

                        while 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == val:
                            segment.append((nr, nc))
                            visited.add((nr, nc))
                            nr, nc = nr + dr, nc + dc

                        # Only extend segments with 2+ cells
                        if len(segment) >= 2:
                            # Determine perpendicular directions
                            if dr == 1 and dc == 1:  # SE segment
                                perp1 = (-1, 1)  # NE
                                perp2 = (1, -1)  # SW
                            else:  # SW segment (dr == 1, dc == -1)
                                perp1 = (-1, -1)  # NW
                                perp2 = (1, 1)  # SE

                            # Check which perpendicular direction to extend from first endpoint
                            r0, c0 = segment[0]
                            nr, nc = r0 + perp1[0], c0 + perp1[1]
                            can_extend_perp1 = (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in [0, 1])

                            nr, nc = r0 + perp2[0], c0 + perp2[1]
                            can_extend_perp2 = (0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] in [0, 1])

                            # Choose direction to extend (prefer perp1 if available)
                            if can_extend_perp1:
                                extend_dir = perp1
                            elif can_extend_perp2:
                                extend_dir = perp2
                            else:
                                continue  # Can't extend

                            # Extend from both endpoints in the chosen perpendicular direction
                            for r_start, c_start in [segment[0], segment[-1]]:
                                nr, nc = r_start + extend_dir[0], c_start + extend_dir[1]
                                while 0 <= nr < rows and 0 <= nc < cols and result[nr][nc] in [0, 1]:
                                    result[nr][nc] = val
                                    nr, nc = nr + extend_dir[0], nc + extend_dir[1]

    return result
