def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Grid is divided by a column of 5s (separator) into left and right sections
    2. Rectangular regions containing 0s in left section move rightward toward separator
    3. Original positions get filled with 6s
    4. If transformed row ends with "6,0" pattern before separator, right section becomes all 2s

    Procedure:
    1. Find separator column (all 5s)
    2. Identify connected rectangular regions of 0s
    3. Move these regions rightward within left section
    4. Apply right section rule based on ending pattern
    """

    rows = len(grid)
    cols = len(grid[0])

    # Find separator column (all 5s)
    separator_col = -1
    for col in range(cols):
        if all(grid[row][col] == 5 for row in range(rows)):
            separator_col = col
            break

    if separator_col == -1:
        return grid

    # Create result grid
    result = [row[:] for row in grid]

    # Identify rectangular regions of 0s by finding bounding boxes
    visited = set()
    regions = []

    for r in range(rows):
        for c in range(separator_col):  # Only check left section
            if grid[r][c] == 0 and (r, c) not in visited:
                # Found a new region, get its bounding box
                min_r, max_r = r, r
                min_c, max_c = c, c

                # BFS to find all connected 0s and their bounding box
                queue = [(r, c)]
                region_cells = set()

                while queue:
                    cr, cc = queue.pop(0)
                    if (
                        (cr, cc) in visited
                        or cr < 0
                        or cr >= rows
                        or cc < 0
                        or cc >= separator_col
                    ):
                        continue
                    if grid[cr][cc] != 0:
                        continue

                    visited.add((cr, cc))
                    region_cells.add((cr, cc))
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)

                    # Add neighbors
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        queue.append((cr + dr, cc + dc))

                if region_cells:
                    regions.append((min_r, max_r, min_c, max_c, region_cells))

    # Move each region rightward
    for min_r, max_r, min_c, max_c, region_cells in regions:
        region_width = max_c - min_c + 1
        region_height = max_r - min_r + 1

        # Calculate target position (move toward separator)
        target_c = separator_col - region_width
        if target_c < 0:
            target_c = 0

        # Extract the rectangular region pattern
        region_pattern = []
        for r in range(min_r, max_r + 1):
            row_pattern = []
            for c in range(min_c, max_c + 1):
                row_pattern.append(grid[r][c])
            region_pattern.append(row_pattern)

        # Clear original region
        for r, c in region_cells:
            result[r][c] = 6

        # Place region at target position
        for r in range(region_height):
            for c in range(region_width):
                if min_r + r < rows and target_c + c < separator_col:
                    result[min_r + r][target_c + c] = region_pattern[r][c]

    # Apply right section rule
    for row in range(rows):
        if separator_col >= 2:
            last_two = result[row][separator_col - 2 : separator_col]
            if last_two == [6, 0]:
                # Make right section all 2s
                for col in range(separator_col + 1, cols):
                    result[row][col] = 2

    return result
