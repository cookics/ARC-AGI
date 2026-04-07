def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has hollow rectangles with colored borders and background interior
    2. Input also has scattered patches of various colors
    3. Output keeps only the rectangles, removes scattered patches
    4. For each rectangle, dots are added inside based on patches of the same color outside
    5. Number of dots = number of connected components of rectangle's color outside the rectangle
    6. Dots are placed in middle row of interior, at specific column positions

    Procedure:
    1. Find background color (most frequent)
    2. Identify all hollow rectangles
    3. For each rectangle, count connected components of its color outside
    4. Draw rectangles with appropriate number of dots inside
    5. Fill rest with background
    """

    def is_valid_rectangle(r1, c1, r2, c2, color):
        """Check if coordinates form a valid hollow rectangle"""
        # Check borders
        for c in range(c1, c2 + 1):
            if grid[r1][c] != color or grid[r2][c] != color:
                return False
        for r in range(r1, r2 + 1):
            if grid[r][c1] != color or grid[r][c2] != color:
                return False
        # Check interior is background
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                if grid[r][c] != background:
                    return False
        return True

    def bfs(visited, start_r, start_c, color):
        """BFS to mark all cells in connected component"""
        queue = [(start_r, start_c)]
        visited[start_r][start_c] = True

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and not visited[nr][nc]
                    and grid[nr][nc] == color
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))

    def count_outside_components(r1, c1, r2, c2, color):
        """Count connected components of given color outside the rectangle"""
        visited = [[False] * cols for _ in range(rows)]

        # Mark rectangle cells as visited
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                visited[r][c] = True

        count = 0
        for r in range(rows):
            for c in range(cols):
                if not visited[r][c] and grid[r][c] == color:
                    # Found new component, do BFS
                    bfs(visited, r, c, color)
                    count += 1

        return count

    def calculate_dot_positions(n, interior_width):
        """Calculate positions for n dots in interior of given width"""
        # Based on observations:
        # Dots are placed with spacing 2, always ending at position 7
        # Formula: end = 7, start = 7 - 2*(n-1)
        # For n=1: [7], n=2: [5,7], n=3: [3,5,7], n=4: [1,3,5,7]
        end = 7
        start = end - 2 * (n - 1)

        positions = []
        for i in range(n):
            pos = start + 2 * i
            if 0 <= pos < interior_width:
                positions.append(pos)

        return positions

    rows, cols = len(grid), len(grid[0])

    # Find background color (most frequent)
    color_count = {}
    for r in range(rows):
        for c in range(cols):
            color_count[grid[r][c]] = color_count.get(grid[r][c], 0) + 1
    background = max(color_count, key=color_count.get)

    # Find all hollow rectangles
    rectangles = []
    for r1 in range(rows - 2):
        for c1 in range(cols - 2):
            if grid[r1][c1] != background:
                color = grid[r1][c1]

                for r2 in range(r1 + 2, rows):
                    for c2 in range(c1 + 2, cols):
                        if is_valid_rectangle(r1, c1, r2, c2, color):
                            rectangles.append((r1, c1, r2, c2, color))

    # Remove duplicates and overlaps
    rectangles = list(set(rectangles))
    rectangles.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    final_rectangles = []
    for rect in rectangles:
        r1, c1, r2, c2, color = rect
        overlaps = False
        for existing in final_rectangles:
            er1, ec1, er2, ec2, _ = existing
            if not (r2 < er1 or r1 > er2 or c2 < ec1 or c1 > ec2):
                overlaps = True
                break
        if not overlaps:
            final_rectangles.append(rect)

    # Create result filled with background
    result = [[background] * cols for _ in range(rows)]

    # Process each rectangle
    for r1, c1, r2, c2, color in final_rectangles:
        # Count connected components of this color outside this rectangle
        num_components = count_outside_components(r1, c1, r2, c2, color)

        # Draw rectangle border
        for c in range(c1, c2 + 1):
            result[r1][c] = color  # Top border
            result[r2][c] = color  # Bottom border
        for r in range(r1, r2 + 1):
            result[r][c1] = color  # Left border
            result[r][c2] = color  # Right border

        # Add dots in middle row based on number of components
        if num_components > 0:
            interior_height = r2 - r1 - 1
            if interior_height > 0:
                middle_row = r1 + 1 + interior_height // 2
                interior_width = c2 - c1 - 1

                # Calculate dot positions
                dot_positions = calculate_dot_positions(num_components, interior_width)

                # Place dots
                for pos in dot_positions:
                    result[middle_row][c1 + 1 + pos] = color

    return result
