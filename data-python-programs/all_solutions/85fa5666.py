def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 2x2 blocks of 2s that remain unchanged in output
    2. Each 2x2 block is surrounded by a rectangle of 4 colored cells (non-zero, non-2)
    3. Each corner of the rectangle shoots a diagonal line outward from the rectangle
    4. The value shot from each corner is swapped with an adjacent corner's value

    Procedure:
    1. Find all 2x2 blocks of 2s in the grid
    2. For each block at rows (r, r+1) and cols (c, c+1):
       - Rectangle corners are at (r-1, c-1), (r-1, c+2), (r+2, c-1), (r+2, c+2)
       - Top-left shoots diagonal up-left with bottom-left's value
       - Top-right shoots diagonal up-right with top-left's value
       - Bottom-left shoots diagonal down-left with bottom-right's value
       - Bottom-right shoots diagonal down-right with top-right's value
    3. Each diagonal continues until it goes out of bounds
    """
    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # First pass: collect all rectangle corners from all blocks
    all_corners = set()
    rectangles = []

    for r in range(rows - 1):
        for c in range(cols - 1):
            if (grid[r][c] == 2 and grid[r][c+1] == 2 and
                grid[r+1][c] == 2 and grid[r+1][c+1] == 2):

                # Rectangle corners around this 2x2 block
                r_tl, c_tl = r - 1, c - 1
                r_tr, c_tr = r - 1, c + 2
                r_bl, c_bl = r + 2, c - 1
                r_br, c_br = r + 2, c + 2

                # Check if all corners are within bounds
                if (0 <= r_tl < rows and 0 <= c_tl < cols and
                    0 <= r_tr < rows and 0 <= c_tr < cols and
                    0 <= r_bl < rows and 0 <= c_bl < cols and
                    0 <= r_br < rows and 0 <= c_br < cols):

                    # Get values at corners
                    v_tl = grid[r_tl][c_tl]
                    v_tr = grid[r_tr][c_tr]
                    v_bl = grid[r_bl][c_bl]
                    v_br = grid[r_br][c_br]

                    # Add to collections
                    all_corners.update([(r_tl, c_tl), (r_tr, c_tr), (r_bl, c_bl), (r_br, c_br)])
                    rectangles.append(((r_tl, c_tl, v_tl), (r_tr, c_tr, v_tr),
                                      (r_bl, c_bl, v_bl), (r_br, c_br, v_br)))

    # Second pass: draw all diagonals
    for (r_tl, c_tl, v_tl), (r_tr, c_tr, v_tr), (r_bl, c_bl, v_bl), (r_br, c_br, v_br) in rectangles:
        # From top-left, draw diagonal up-left with bottom-left's value (upward - no adjacency check)
        dr, dc = r_tl, c_tl
        while 0 <= dr < rows and 0 <= dc < cols:
            if grid[dr][dc] == 2 and (dr, dc) not in all_corners:
                break
            result[dr][dc] = v_bl
            dr -= 1
            dc -= 1

        # From top-right, draw diagonal up-right with top-left's value (upward - no adjacency check)
        dr, dc = r_tr, c_tr
        while 0 <= dr < rows and 0 <= dc < cols:
            if grid[dr][dc] == 2 and (dr, dc) not in all_corners:
                break
            result[dr][dc] = v_tl
            dr -= 1
            dc += 1

        # From bottom-left, draw diagonal down-left with bottom-right's value (downward - check right adjacency)
        dr, dc = r_bl, c_bl
        while 0 <= dr < rows and 0 <= dc < cols:
            if grid[dr][dc] == 2 and (dr, dc) not in all_corners:
                break
            # For down-left diagonal, stop if 2 is to the RIGHT (moving away from it)
            if (dr, dc) not in all_corners:
                if 0 <= dc + 1 < cols and grid[dr][dc + 1] == 2:
                    break
            result[dr][dc] = v_br
            dr += 1
            dc -= 1

        # From bottom-right, draw diagonal down-right with top-right's value (downward - no adjacency check)
        dr, dc = r_br, c_br
        while 0 <= dr < rows and 0 <= dc < cols:
            if grid[dr][dc] == 2 and (dr, dc) not in all_corners:
                break
            result[dr][dc] = v_tr
            dr += 1
            dc += 1

    return result
