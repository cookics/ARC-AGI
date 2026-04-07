def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with vertical separator (column of 6s)
    2. Output is the transformed right section (canvas after separator)
    3. Left section contains frames with patterns to be tiled onto canvas
    4. Pattern tiles with double-period (3x3 pattern tiles every 6 rows/cols)
    5. Pattern appears in "second half" of each tile period
    6. Canvas values are preserved where pattern has 0

    Procedure:
    1. Find vertical separator column (all 6s)
    2. Extract left (pattern) and right (canvas) sections
    3. Find frames in left section (rectangles bordered by 6s)
    4. Extract pattern from first top frame
    5. Tile pattern onto canvas with proper offset and period
    6. Return transformed canvas
    """

    height = len(grid)
    width = len(grid[0])

    # Find vertical separator
    sep_col = -1
    for col in range(width):
        if all(grid[r][col] == 6 for r in range(height)):
            sep_col = col
            break

    if sep_col == -1:
        return grid

    # Extract sections
    left = [row[:sep_col] for row in grid]
    canvas = [row[sep_col+1:] for row in grid]

    if not left or not left[0]:
        return canvas

    # Find horizontal separator in left
    horiz_sep = -1
    for r in range(height):
        if len(left[0]) > 0 and all(left[r][c] == 6 for c in range(len(left[0]))):
            horiz_sep = r
            break

    # Extract first frame pattern from top section
    pattern = None
    search_end = horiz_sep if horiz_sep != -1 else height

    for r in range(search_end - 2):
        if pattern:
            break
        for c in range(len(left[0]) - 2):
            if left[r][c] == 6:
                # Try to find complete frame
                for r2 in range(r + 2, min(search_end, r + 8)):
                    for c2 in range(c + 2, min(len(left[0]), c + 8)):
                        # Check if valid frame
                        valid = all(left[r][cc] == 6 and left[r2][cc] == 6
                                   for cc in range(c, min(c2+1, len(left[0]))))
                        if valid:
                            valid = all(left[rr][c] == 6 and left[rr][c2] == 6
                                       for rr in range(r, r2+1))

                        if valid and r2 > r+1 and c2 > c+1:
                            # Extract interior
                            pat = [left[rr][c+1:c2] for rr in range(r+1, r2)]
                            if pat and pat[0] and any(v not in [0,6] for row in pat for v in row):
                                pattern = pat
                                break
                    if pattern:
                        break
                if pattern:
                    break

    # Apply pattern tiling to canvas
    result = [row[:] for row in canvas]

    if not pattern:
        return result

    pat_h = len(pattern)
    pat_w = len(pattern[0])

    # Tile pattern with offset
    # Pattern appears every (2 * pattern_size) with offset
    for r in range(len(result)):
        for c in range(len(result[0])):
            # Simple mod tiling - pattern repeats with its own dimensions
            pr = r % pat_h
            pc = c % pat_w

            pat_val = pattern[pr][pc]

            # Replace canvas value with pattern value if pattern is non-zero/non-6
            if pat_val not in [0, 6]:
                result[r][c] = pat_val

    return result
