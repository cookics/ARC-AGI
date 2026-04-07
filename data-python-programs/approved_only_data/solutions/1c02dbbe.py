def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with a rectangular region filled with 5s.
    2. Marker colors (non-zero, non-5 values) are positioned around the perimeter of the 5s region.
    3. Output divides the 5s region into sections based on marker positions.
    4. Each marker color takes over a portion of the region, with boundaries determined by marker extents.
    5. The division creates quadrant-like sections where each marker color fills its associated area.

    Procedure:
    1. Find the bounding rectangle of all 5s in the grid.
    2. Identify all marker colors and their positions around the 5s region.
    3. Determine which quadrant each marker color belongs to based on its position relative to the 5s center.
    4. Calculate the exact boundaries each color should control within the 5s area.
    5. Fill the result grid with appropriate colors based on the determined boundaries.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find bounding box of 5s
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    assert min_row != rows, "No 5s found in grid"

    # Find marker colors and their positions
    markers = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and grid[r][c] != 5:
                color = grid[r][c]
                if color not in markers:
                    markers[color] = []
                markers[color].append((r, c))

    # Analyze each color to determine its quadrant and influence boundaries
    color_regions = {}

    for color, positions in markers.items():
        # Determine primary quadrant based on position relative to 5s center
        center_row = (min_row + max_row) / 2
        center_col = (min_col + max_col) / 2

        # Categorize positions
        top_left = sum(1 for r, c in positions if r < center_row and c < center_col)
        top_right = sum(1 for r, c in positions if r < center_row and c > center_col)
        bottom_left = sum(1 for r, c in positions if r > center_row and c < center_col)
        bottom_right = sum(1 for r, c in positions if r > center_row and c > center_col)

        # Also count edge positions
        above = sum(1 for r, c in positions if r < min_row)
        below = sum(1 for r, c in positions if r > max_row)
        left = sum(1 for r, c in positions if c < min_col)
        right = sum(1 for r, c in positions if c > max_col)

        # Determine the primary quadrant
        quadrant_scores = {
            "top_left": top_left + above + left,
            "top_right": top_right + above + right,
            "bottom_left": bottom_left + below + left,
            "bottom_right": bottom_right + below + right,
        }

        primary_quadrant = max(quadrant_scores, key=quadrant_scores.get)

        # Find the extent of this color's influence
        max_row_extent = max(r for r, c in positions)
        min_row_extent = min(r for r, c in positions)
        max_col_extent = max(c for r, c in positions)
        min_col_extent = min(c for r, c in positions)

        color_regions[color] = {
            "quadrant": primary_quadrant,
            "row_extent": (min_row_extent, max_row_extent),
            "col_extent": (min_col_extent, max_col_extent),
        }

    # Determine specific boundaries for each color based on their influence
    color_boundaries = {}

    for color, info in color_regions.items():
        min_r, max_r = info["row_extent"]
        min_c, max_c = info["col_extent"]
        quadrant = info["quadrant"]

        # Determine the exact region this color should control within the 5s area
        if quadrant == "top_left":
            # Extend from top-left of 5s region
            color_boundaries[color] = {
                "min_row": min_row,
                "max_row": max_r if max_r >= min_row else (min_row + max_row) // 2,
                "min_col": min_col,
                "max_col": max_c if max_c >= min_col else (min_col + max_col) // 2,
            }
        elif quadrant == "top_right":
            # Extend from top-right of 5s region
            color_boundaries[color] = {
                "min_row": min_row,
                "max_row": max_r if max_r >= min_row else (min_row + max_row) // 2,
                "min_col": min_c if min_c <= max_col else (min_col + max_col) // 2,
                "max_col": max_col,
            }
        elif quadrant == "bottom_left":
            # Extend from bottom-left of 5s region
            color_boundaries[color] = {
                "min_row": min_r if min_r <= max_row else (min_row + max_row) // 2,
                "max_row": max_row,
                "min_col": min_col,
                "max_col": max_c if max_c >= min_col else (min_col + max_col) // 2,
            }
        elif quadrant == "bottom_right":
            # Extend from bottom-right of 5s region
            color_boundaries[color] = {
                "min_row": min_r if min_r <= max_row else (min_row + max_row) // 2,
                "max_row": max_row,
                "min_col": min_c if min_c <= max_col else (min_col + max_col) // 2,
                "max_col": max_col,
            }

    # Fill the result grid
    for r in range(rows):
        for c in range(cols):
            if min_row <= r <= max_row and min_col <= c <= max_col:
                cell_color = 5  # default

                # Check each color to see if this cell falls in its region
                for color, bounds in color_boundaries.items():
                    if (
                        bounds["min_row"] <= r <= bounds["max_row"]
                        and bounds["min_col"] <= c <= bounds["max_col"]
                    ):
                        cell_color = color
                        break

                result[r][c] = cell_color
            else:
                result[r][c] = 0

    return result
