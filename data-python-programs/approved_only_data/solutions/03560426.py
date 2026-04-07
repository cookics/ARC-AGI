def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with 0s representing empty space and non-zero values representing colored blocks.
    2. Input contains colored rectangular blocks scattered mostly in the bottom rows of the grid.
    3. Output rearranges these blocks in a diagonal chain pattern starting from the top-left corner.
    4. Blocks are connected corner-to-corner in ascending order of their leftmost column position in the input.
    5. Later blocks can overwrite earlier blocks at shared corner positions.

    Procedure:
    1. Scan the grid to identify all colored rectangular blocks using flood fill algorithm.
    2. Extract each block's position, dimensions, and color information.
    3. Sort blocks by their leftmost column position in ascending order.
    4. Place the first block at position (0,0) in the output grid.
    5. For each subsequent block, place its top-left corner at the previous block's bottom-right corner.
    6. Fill the result grid with each block's color at their new positions.
    """

    rows, cols = len(grid), len(grid[0])

    # Find all colored blocks
    visited = [[False] * cols for _ in range(rows)]
    blocks = []

    def flood_fill(r, c, color):
        """Find the bounding box of a colored region"""
        stack = [(r, c)]
        cells = []

        while stack:
            cr, cc = stack.pop()
            if (
                cr < 0
                or cr >= rows
                or cc < 0
                or cc >= cols
                or visited[cr][cc]
                or grid[cr][cc] != color
            ):
                continue

            visited[cr][cc] = True
            cells.append((cr, cc))

            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((cr + dr, cc + dc))

        if cells:
            min_r = min(r for r, c in cells)
            max_r = max(r for r, c in cells)
            min_c = min(c for r, c in cells)
            max_c = max(c for r, c in cells)
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            return (min_r, min_c, height, width, color)
        return None

    # Find all blocks
    for r in range(rows):
        for c in range(cols):
            if not visited[r][c] and grid[r][c] != 0:
                block = flood_fill(r, c, grid[r][c])
                if block:
                    blocks.append(block)

    # Sort blocks by leftmost column position
    blocks.sort(key=lambda x: x[1])  # Sort by min_c

    # Verify we found blocks correctly
    assert len(blocks) > 0, "Should find at least one block"

    # Create output grid
    result = [[0] * cols for _ in range(rows)]

    # Place blocks in chain pattern
    current_row, current_col = 0, 0

    for i, (_, _, height, width, color) in enumerate(blocks):
        # Place current block
        for r in range(height):
            for c in range(width):
                if current_row + r < rows and current_col + c < cols:
                    result[current_row + r][current_col + c] = color

        # Next block's top-left goes at current block's bottom-right
        current_row = current_row + height - 1
        current_col = current_col + width - 1

    return result
