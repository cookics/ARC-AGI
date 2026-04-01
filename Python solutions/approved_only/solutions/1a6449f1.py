def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid containing integers (including 0).
    2. Output is a smaller 2D grid extracted from the input.
    3. The input contains rectangular regions bounded by lines of the same non-zero number.
    4. These boundary lines form complete rectangles with top, bottom, left, and right edges.
    5. The task is to extract the interior content of one of these rectangles.
    6. The boundary lines themselves are excluded from the output.

    Procedure:
    1. Find all unique non-zero numbers that could form rectangular boundaries.
    2. For each number, detect horizontal and vertical line segments.
    3. Combine line segments to form complete rectangles.
    4. Extract the interior content of each valid rectangle.
    5. Return the interior with the largest area as the result.
    """

    def find_rectangles_bounded_by(grid, num):
        """Find complete rectangles bounded by lines of the given number"""
        height, width = len(grid), len(grid[0])
        rectangles = []

        # Find horizontal line segments
        h_lines = []
        for r in range(height):
            segments = []
            start = None
            for c in range(width):
                if grid[r][c] == num:
                    if start is None:
                        start = c
                else:
                    if start is not None:
                        if c - start >= 3:  # Minimum length for a meaningful boundary
                            segments.append((start, c - 1))
                        start = None
            if start is not None and width - start >= 3:
                segments.append((start, width - 1))

            for seg in segments:
                h_lines.append((r, seg[0], seg[1]))

        # Find vertical line segments
        v_lines = []
        for c in range(width):
            segments = []
            start = None
            for r in range(height):
                if grid[r][c] == num:
                    if start is None:
                        start = r
                else:
                    if start is not None:
                        if r - start >= 3:
                            segments.append((start, r - 1))
                        start = None
            if start is not None and height - start >= 3:
                segments.append((start, height - 1))

            for seg in segments:
                v_lines.append((c, seg[0], seg[1]))

        # Form rectangles from horizontal and vertical lines
        for i, h1 in enumerate(h_lines):
            for j, h2 in enumerate(h_lines):
                if i >= j:
                    continue
                r1, c1_start, c1_end = h1
                r2, c2_start, c2_end = h2

                # Check if they could form top/bottom of rectangle
                if (
                    r2 > r1
                    and c1_start == c2_start
                    and c1_end == c2_end
                    and r2 - r1 >= 2
                    and c1_end - c1_start >= 2
                ):
                    # Look for matching vertical lines
                    left_found = False
                    right_found = False

                    for v_line in v_lines:
                        c_v, r_v_start, r_v_end = v_line
                        if c_v == c1_start and r_v_start <= r1 and r_v_end >= r2:
                            left_found = True
                        if c_v == c1_end and r_v_start <= r1 and r_v_end >= r2:
                            right_found = True

                    if left_found and right_found:
                        rectangles.append((r1, r2, c1_start, c1_end))

        return rectangles

    def extract_interior(grid, rect):
        """Extract the interior content of a rectangle (excluding boundaries)"""
        r1, r2, c1, c2 = rect
        if r2 - r1 <= 2 or c2 - c1 <= 2:
            return None

        interior = []
        for r in range(r1 + 1, r2):
            row = []
            for c in range(c1 + 1, c2):
                row.append(grid[r][c])
            interior.append(row)
        return interior

    height, width = len(grid), len(grid[0])

    # Find unique non-zero numbers that could form boundaries
    unique_nums = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                unique_nums.add(cell)

    # Find all valid rectangles and their interiors
    all_interiors = []
    for num in unique_nums:
        rectangles = find_rectangles_bounded_by(grid, num)
        for rect in rectangles:
            interior = extract_interior(grid, rect)
            if interior and len(interior) > 0 and len(interior[0]) > 0:
                all_interiors.append(interior)

    # Return the interior with the largest area (most likely the target)
    if all_interiors:
        return max(all_interiors, key=lambda x: len(x) * len(x[0]))

    return []
