def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains rectangular frames made of 0s with colored pixels inside
    2. Each colored pixel near a frame corner gets "reflected" outside that corner
    3. Distance 1 from corner: L-shape pattern (corner + 2 adjacent cells)
    4. Distance 2+ from corner: 2x2 block + diagonal path to corner

    Procedure:
    1. Find background color (most common non-zero value)
    2. For each colored cell (not background, not 0):
       - Find nearest 0 in each of 4 directions
       - If has 0s forming a corner within distance 2, create reflection
       - Pattern depends on Manhattan distance to corner
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    from collections import Counter
    flat = [cell for row in grid for cell in row]
    background = Counter(flat).most_common(1)[0][0]

    for r in range(rows):
        for c in range(cols):
            color = grid[r][c]
            if color != background and color != 0:
                # Find nearest 0 in each direction
                up_r, down_r, left_c, right_c = -1, -1, -1, -1
                up_dist, down_dist, left_dist, right_dist = 999, 999, 999, 999

                for rr in range(r - 1, -1, -1):
                    if grid[rr][c] == 0:
                        up_r, up_dist = rr, r - rr
                        break

                for rr in range(r + 1, rows):
                    if grid[rr][c] == 0:
                        down_r, down_dist = rr, rr - r
                        break

                for cc in range(c - 1, -1, -1):
                    if grid[r][cc] == 0:
                        left_c, left_dist = cc, c - cc
                        break

                for cc in range(c + 1, cols):
                    if grid[r][cc] == 0:
                        right_c, right_dist = cc, cc - c
                        break

                # Check for corners: need 0s in two perpendicular directions
                corners = []
                if up_r >= 0 and left_c >= 0 and up_dist <= 2 and left_dist <= 2:
                    if grid[up_r][left_c] == 0:
                        corners.append(('TL', up_dist + left_dist, up_r, left_c))
                if up_r >= 0 and right_c >= 0 and up_dist <= 2 and right_dist <= 2:
                    if grid[up_r][right_c] == 0:
                        corners.append(('TR', up_dist + right_dist, up_r, right_c))
                if down_r >= 0 and left_c >= 0 and down_dist <= 2 and left_dist <= 2:
                    if grid[down_r][left_c] == 0:
                        corners.append(('BL', down_dist + left_dist, down_r, left_c))
                if down_r >= 0 and right_c >= 0 and down_dist <= 2 and right_dist <= 2:
                    if grid[down_r][right_c] == 0:
                        corners.append(('BR', down_dist + right_dist, down_r, right_c))

                if not corners:
                    continue

                # Choose nearest corner
                corner_type, total_dist, corner_r, corner_c = min(corners, key=lambda x: x[1])

                # Determine distances
                dist_r = abs(r - corner_r)
                dist_c = abs(c - corner_c)

                # Check if the value is already part of a reflected pattern
                # Count instances of this color in a 3x3 around the corner
                color_count = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = corner_r + dr, corner_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if grid[nr][nc] == color:
                                color_count += 1

                # If there are already 2+ instances of this color around the corner,
                # it's likely already reflected - use L-shape
                use_l_shape = (color_count >= 2)

                # Place reflection pattern based on distance and corner type
                if corner_type == 'TL':
                    if dist_r == 1 and dist_c == 1:
                        if use_l_shape:
                            # L-shape: corner + two adjacent cells
                            result[corner_r][corner_c] = color
                            if corner_r - 1 >= 0:
                                result[corner_r - 1][corner_c] = color
                            if corner_c - 1 >= 0:
                                result[corner_r][corner_c - 1] = color
                        else:
                            # 3x3 cross pattern around corner (skip opposite diagonals and corner)
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    # Skip opposite diagonal corners
                                    if (dr, dc) == (-1, -1) or (dr, dc) == (1, 1):
                                        continue
                                    # Skip the corner itself (leave as 0)
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = corner_r + dr, corner_c + dc
                                    if 0 <= nr < rows and 0 <= nc < cols:
                                        result[nr][nc] = color
                    else:
                        # 2x2 block outside corner + diagonal
                        for dr in range(2):
                            for dc in range(2):
                                nr, nc = corner_r - 1 + dr, corner_c - 1 + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    result[nr][nc] = color
                        # Diagonal from corner towards value
                        dr, dc = corner_r + 1, corner_c + 1
                        while dr < r and dc < c:
                            result[dr][dc] = color
                            dr += 1
                            dc += 1
                elif corner_type == 'TR':
                    if dist_r == 1 and dist_c == 1:
                        if use_l_shape:
                            # L-shape: corner + two adjacent cells
                            result[corner_r][corner_c] = color
                            if corner_r - 1 >= 0:
                                result[corner_r - 1][corner_c] = color
                            if corner_c + 1 < cols:
                                result[corner_r][corner_c + 1] = color
                        else:
                            # 3x3 cross pattern around corner (skip opposite diagonals and corner)
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    # Skip opposite diagonal corners
                                    if (dr, dc) == (-1, -1) or (dr, dc) == (1, 1):
                                        continue
                                    # Skip the corner itself (leave as 0)
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = corner_r + dr, corner_c + dc
                                    if 0 <= nr < rows and 0 <= nc < cols:
                                        result[nr][nc] = color
                    else:
                        # 2x2 block outside corner + diagonal
                        for dr in range(2):
                            for dc in range(2):
                                nr, nc = corner_r - 1 + dr, corner_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    result[nr][nc] = color
                        # Diagonal from corner towards value
                        dr, dc = corner_r + 1, corner_c - 1
                        while dr < r and dc > c:
                            result[dr][dc] = color
                            dr += 1
                            dc -= 1
                elif corner_type == 'BL':
                    if dist_r == 1 and dist_c == 1:
                        if use_l_shape:
                            # L-shape: corner + two adjacent cells
                            result[corner_r][corner_c] = color
                            if corner_r + 1 < rows:
                                result[corner_r + 1][corner_c] = color
                            if corner_c - 1 >= 0:
                                result[corner_r][corner_c - 1] = color
                        else:
                            # 3x3 cross pattern around corner (skip opposite diagonals and corner)
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    # Skip opposite diagonal corners
                                    if (dr, dc) == (-1, -1) or (dr, dc) == (1, 1):
                                        continue
                                    # Skip the corner itself (leave as 0)
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = corner_r + dr, corner_c + dc
                                    if 0 <= nr < rows and 0 <= nc < cols:
                                        result[nr][nc] = color
                    else:
                        # 2x2 block outside corner + diagonal
                        for dr in range(2):
                            for dc in range(2):
                                nr, nc = corner_r + dr, corner_c - 1 + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    result[nr][nc] = color
                        # Diagonal from corner towards value
                        dr, dc = corner_r - 1, corner_c + 1
                        while dr > r and dc < c:
                            result[dr][dc] = color
                            dr -= 1
                            dc += 1
                else:  # BR
                    if dist_r == 1 and dist_c == 1:
                        if use_l_shape:
                            # L-shape: corner + two adjacent cells
                            result[corner_r][corner_c] = color
                            if corner_r + 1 < rows:
                                result[corner_r + 1][corner_c] = color
                            if corner_c + 1 < cols:
                                result[corner_r][corner_c + 1] = color
                        else:
                            # 3x3 cross pattern around corner (skip opposite diagonals and corner)
                            for dr in range(-1, 2):
                                for dc in range(-1, 2):
                                    # Skip opposite diagonal corners
                                    if (dr, dc) == (-1, -1) or (dr, dc) == (1, 1):
                                        continue
                                    # Skip the corner itself (leave as 0)
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = corner_r + dr, corner_c + dc
                                    if 0 <= nr < rows and 0 <= nc < cols:
                                        result[nr][nc] = color
                    else:
                        # 2x2 block outside corner + diagonal
                        for dr in range(2):
                            for dc in range(2):
                                nr, nc = corner_r + dr, corner_c + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    result[nr][nc] = color
                        # Diagonal from corner towards value
                        dr, dc = corner_r - 1, corner_c - 1
                        while dr > r and dc > c:
                            result[dr][dc] = color
                            dr -= 1
                            dc -= 1

    return result
