def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 4x4 colored regions (solid blocks or hollow rectangles).
    2. Hollow rectangles have colored borders and 0s in the 2x2 interior.
    3. Output contains only hollow rectangles, ignoring solid blocks.
    4. Hollow rectangles are arranged horizontally or vertically based on their spatial distribution.
    5. If there are 3+ hollow rectangles or minimal vertical gaps, arrange horizontally.
    6. If there are 2 hollow rectangles with significant vertical gap, stack vertically.

    Procedure:
    1. Scan the grid to find all 4x4 hollow rectangles (check border color and interior zeros).
    2. Extract position (row, col) and content for each hollow rectangle.
    3. Determine arrangement direction based on count and vertical gaps.
    4. Sort by row (vertical) or column (horizontal) position.
    5. Assemble output by stacking or concatenating rectangles.
    """

    def is_hollow_rectangle(start_row, start_col):
        """Check if 4x4 region starting at (start_row, start_col) is a hollow rectangle"""
        if start_row + 4 > len(grid) or start_col + 4 > len(grid[0]):
            return False, 0

        # Extract 4x4 region
        region = []
        for i in range(4):
            row = []
            for j in range(4):
                row.append(grid[start_row + i][start_col + j])
            region.append(row)

        # Get the color (should be non-zero for border)
        color = region[0][0]
        if color == 0:
            return False, 0

        # Check if all border cells have the same color
        for i in range(4):
            if region[0][i] != color or region[3][i] != color:  # top and bottom
                return False, 0
            if region[i][0] != color or region[i][3] != color:  # left and right
                return False, 0

        # Check if interior cells are empty (0)
        for i in range(1, 3):
            for j in range(1, 3):
                if region[i][j] != 0:
                    return False, 0

        return True, color

    # Find all hollow rectangles
    hollow_rectangles = []

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            is_hollow, color = is_hollow_rectangle(i, j)
            if is_hollow:
                # Extract the 4x4 hollow rectangle
                rect = []
                for di in range(4):
                    row = []
                    for dj in range(4):
                        row.append(grid[i + di][j + dj])
                    rect.append(row)
                hollow_rectangles.append((i, j, rect))

    if not hollow_rectangles:
        return []

    # Decide between vertical stacking and horizontal arrangement
    # Sort by row position first to check for vertical gaps
    hollow_rectangles.sort(key=lambda x: x[0])

    # Simple rule: if 3+ rectangles, always horizontal. If 2, check vertical gaps.
    if len(hollow_rectangles) >= 3:
        has_vertical_gaps = False
    else:
        # Check if there are significant vertical gaps between rectangles
        has_vertical_gaps = False
        if len(hollow_rectangles) == 2:
            rect1, rect2 = hollow_rectangles[0], hollow_rectangles[1]
            gap = rect2[0] - (
                rect1[0] + 3
            )  # gap between bottom of first and top of second
            has_vertical_gaps = gap > 1

    if has_vertical_gaps:
        # Stack vertically (keep sorted by row position)
        result = []
        for _, _, rect in hollow_rectangles:
            result.extend(rect)
    else:
        # Arrange horizontally (sort by column position)
        hollow_rectangles.sort(key=lambda x: x[1])
        result = []
        for row_idx in range(4):
            result_row = []
            for _, _, rect in hollow_rectangles:
                result_row.extend(rect[row_idx])
            result.append(result_row)

    return result
