def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a "1" marker in the top row and structures made of "2"s forming U-shaped or rectangular enclosures
    2. The "1" indicates where to apply a filling operation based on the structures below
    3. U-shaped chambers have left wall, right wall, and bottom connection made of "2"s
    4. If marker column falls within a U-chamber interior, fill that chamber with "1"s
    5. If marker is between close vertical walls but no complete chamber, fill vertical space
    6. Otherwise, fill the entire bottom row with "1"s

    Procedure:
    1. Find the position of the "1" marker and remove it from the grid
    2. Identify U-shaped chambers formed by "2"s (left wall, right wall, bottom connection)
    3. If marker column is inside a U-chamber, fill the interior with "1"s
    4. Otherwise, try vertical filling between close walls containing the marker column
    5. If no structure contains the marker, fill the entire bottom row with "1"s
    """

    def check_bottom_connection(twos, left_col, right_col, bottom_row):
        """Check if there's a complete bottom connection between two vertical walls"""
        for c in range(left_col, right_col + 1):
            if (bottom_row, c) not in twos:
                return False
        return True

    def try_fill_u_chambers(result, twos, marker_col, rows, cols):
        """Try to find and fill U-shaped chambers that contain the marker column"""
        vertical_walls = []
        for c in range(cols):
            wall_rows = [r for r in range(rows) if (r, c) in twos]
            if len(wall_rows) >= 3:
                wall_rows.sort()
                continuous = True
                for i in range(len(wall_rows) - 1):
                    if wall_rows[i + 1] - wall_rows[i] > 1:
                        continuous = False
                        break
                if continuous:
                    vertical_walls.append((c, min(wall_rows), max(wall_rows)))

        for i in range(len(vertical_walls)):
            for j in range(i + 1, len(vertical_walls)):
                left_col, left_min, left_max = vertical_walls[i]
                right_col, right_min, right_max = vertical_walls[j]

                if left_col < marker_col < right_col:
                    bottom_row = max(left_max, right_max)
                    has_bottom = check_bottom_connection(twos, left_col, right_col, bottom_row)

                    if has_bottom:
                        top_row = max(left_min, right_min)
                        for r in range(top_row, bottom_row):
                            for c in range(left_col + 1, right_col):
                                if result[r][c] == 0:
                                    result[r][c] = 1
                        return True
        return False

    def has_vertical_wall(twos, col, rows):
        """Check if a column has a vertical wall (3+ consecutive "2"s)"""
        positions = [r for r in range(rows) if (r, col) in twos]
        if len(positions) < 3:
            return False

        positions.sort()
        consecutive_count = 1
        for i in range(len(positions) - 1):
            if positions[i + 1] == positions[i] + 1:
                consecutive_count += 1
                if consecutive_count >= 3:
                    return True
            else:
                consecutive_count = 1
        return False

    def try_fill_vertical(result, twos, marker_col, rows, cols):
        """Try to fill vertical space between walls"""
        left_wall = None
        right_wall = None

        for c in range(cols):
            if has_vertical_wall(twos, c, rows):
                if c < marker_col:
                    left_wall = c
                elif c > marker_col and right_wall is None:
                    right_wall = c
                    break

        if left_wall is not None and right_wall is not None:
            if right_wall - left_wall <= 3:
                left_rows = [r for r in range(rows) if (r, left_wall) in twos]
                right_rows = [r for r in range(rows) if (r, right_wall) in twos]

                if left_rows and right_rows:
                    min_row = max(min(left_rows), min(right_rows))
                    max_row = min(max(left_rows), max(right_rows))

                    for r in range(min_row, max_row + 1):
                        if result[r][marker_col] == 0:
                            result[r][marker_col] = 1
                    return True
        return False

    def fill_bottom_row(result, rows, cols):
        """Fill the entire bottom row with 1s"""
        for c in range(cols):
            if result[rows - 1][c] == 0:
                result[rows - 1][c] = 1

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    marker_col = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                marker_col = c
                result[r][c] = 0
                break
        if marker_col is not None:
            break

    twos = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                twos.add((r, c))

    filled = try_fill_u_chambers(result, twos, marker_col, rows, cols)

    if not filled:
        filled = try_fill_vertical(result, twos, marker_col, rows, cols)

    if not filled:
        fill_bottom_row(result, rows, cols)

    return result
