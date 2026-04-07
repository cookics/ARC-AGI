def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with values 0, 4, 5, and 7
    2. Output has same dimensions, transforming 7-blocks and original 5s
    3. Grid contains rectangular regions of 7s that need partial conversion to 5s
    4. Conversion depends on dimensions: height=1 and width>3 stays 7s, width>3 converts first row, height>2 converts first 2 columns, else converts all
    5. All original 5s (not converted from 7s) become 4s

    Procedure:
    1. Find all rectangular connected components of 7s using flood fill
    2. For each 7-block, apply dimension-based conversion rules to replace some 7s with 5s
    3. Convert all original 5s (from input) to 4s
    4. Return the transformed grid
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # deep copy

    # Find all 7-regions
    visited = [[False] * cols for _ in range(rows)]
    seven_regions = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 7 and not visited[r][c]:
                # Find the rectangular region starting at (r, c)
                min_r, max_r = r, r
                min_c, max_c = c, c

                # Find bounds of the rectangle
                # First find the extent in the row direction
                while max_c + 1 < cols and grid[r][max_c + 1] == 7:
                    max_c += 1

                # Then find the extent in the column direction
                valid = True
                while max_r + 1 < rows and valid:
                    for col in range(min_c, max_c + 1):
                        if grid[max_r + 1][col] != 7:
                            valid = False
                            break
                    if valid:
                        max_r += 1

                # Mark all cells in this rectangle as visited
                for rr in range(min_r, max_r + 1):
                    for cc in range(min_c, max_c + 1):
                        visited[rr][cc] = True

                seven_regions.append((min_r, max_r, min_c, max_c))

    # Apply conversion rules to each 7-region
    for min_r, max_r, min_c, max_c in seven_regions:
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        if height == 1 and width > 3:
            # No conversion
            continue
        elif width > 3:
            # Convert first row to 5s
            for cc in range(min_c, max_c + 1):
                result[min_r][cc] = 5
        elif height > 2:
            # Convert first 2 columns to 5s
            for rr in range(min_r, max_r + 1):
                for cc in range(min_c, min(min_c + 2, max_c + 1)):
                    result[rr][cc] = 5
        else:
            # Convert entire block to 5s
            for rr in range(min_r, max_r + 1):
                for cc in range(min_c, max_c + 1):
                    result[rr][cc] = 5

    # Convert all remaining 5s to 4s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:  # Original 5s
                result[r][c] = 4

    return result
