def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with hollow rectangular shapes made of various colors
    2. Each hollow rectangle has a uniform frame color on the entire perimeter
    3. Interior of hollow rectangles is filled with 0s
    4. There's a unique marker value that appears exactly once in the grid
    5. Output fills the interior of hollow rectangles with the marker value

    Procedure:
    1. Find the marker value (appears only once in the grid)
    2. Enumerate all possible rectangles
    3. For each rectangle, check if:
       - Interior is all 0s
       - Entire perimeter has uniform non-zero color
    4. Fill valid rectangles with marker
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find marker value (appears exactly once)
    value_count = Counter()
    for r in range(rows):
        for c in range(cols):
            value_count[grid[r][c]] += 1

    marker = None
    for val, count in value_count.items():
        if count == 1 and val != 0:
            marker = val
            break

    if marker is None:
        return result

    # Try all possible rectangles
    for r1 in range(rows):
        for c1 in range(cols):
            for r2 in range(r1 + 2, rows):  # Need at least 3 rows for interior
                for c2 in range(c1 + 2, cols):  # Need at least 3 cols for interior
                    # Get frame color from top-left corner
                    frame_color = grid[r1][c1]
                    if frame_color == 0 or frame_color == marker:
                        continue

                    # Check if entire perimeter has uniform frame color
                    valid_frame = True

                    # Check top edge
                    for c in range(c1, c2 + 1):
                        if grid[r1][c] != frame_color:
                            valid_frame = False
                            break

                    if not valid_frame:
                        continue

                    # Check bottom edge
                    for c in range(c1, c2 + 1):
                        if grid[r2][c] != frame_color:
                            valid_frame = False
                            break

                    if not valid_frame:
                        continue

                    # Check left edge
                    for r in range(r1, r2 + 1):
                        if grid[r][c1] != frame_color:
                            valid_frame = False
                            break

                    if not valid_frame:
                        continue

                    # Check right edge
                    for r in range(r1, r2 + 1):
                        if grid[r][c2] != frame_color:
                            valid_frame = False
                            break

                    if not valid_frame:
                        continue

                    # Check if interior is all 0s
                    interior_all_zero = True
                    for r in range(r1 + 1, r2):
                        for c in range(c1 + 1, c2):
                            if grid[r][c] != 0:
                                interior_all_zero = False
                                break
                        if not interior_all_zero:
                            break

                    if not interior_all_zero:
                        continue

                    # Fill interior with marker (only fill cells that are still 0)
                    for r in range(r1 + 1, r2):
                        for c in range(c1 + 1, c2):
                            if result[r][c] == 0:  # Only fill if not already filled
                                result[r][c] = marker

    return result
