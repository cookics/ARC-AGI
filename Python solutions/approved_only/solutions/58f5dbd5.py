def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has 5x5 solid colored rectangles (large blocks) and small scattered patterns
    2. Each color appears in both a large 5x5 solid block and a small pattern
    3. Output creates 5x5 blocks with borders matching block color
    4. Interior (3x3) is filled with INVERTED small pattern (color↔background swap)
    5. Output blocks arranged in same spatial layout as input large blocks
    6. Output has top and bottom border rows, left and right border columns

    Procedure:
    1. Identify background color (most frequent)
    2. Find all 5x5 solid blocks
    3. For each color with a large block, find small pattern (cells not in large block)
    4. Create output blocks with inverted patterns
    5. Arrange output blocks matching input layout with proper borders
    """
    from collections import Counter

    rows, cols = len(grid), len(grid[0])

    # Identify background color
    all_cells = [cell for row in grid for cell in row]
    background = Counter(all_cells).most_common(1)[0][0]

    # Find all 5x5 solid blocks
    large_blocks = []
    large_block_cells = set()

    for r in range(rows - 4):
        for c in range(cols - 4):
            # Check if this is a 5x5 solid block
            color = grid[r][c]
            if color == background:
                continue

            is_solid = True
            for dr in range(5):
                for dc in range(5):
                    if grid[r + dr][c + dc] != color:
                        is_solid = False
                        break
                if not is_solid:
                    break

            if is_solid:
                # Mark all cells in this block
                for dr in range(5):
                    for dc in range(5):
                        large_block_cells.add((r + dr, c + dc))
                large_blocks.append((color, r, c))

    # For each color with a large block, find the small pattern
    pattern_dict = {}
    colors_with_blocks = set(color for color, _, _ in large_blocks)

    for color in colors_with_blocks:
        # Find all cells of this color NOT in large blocks
        pattern_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == color and (r, c) not in large_block_cells:
                    pattern_cells.append((r, c))

        if pattern_cells:
            # Extract bounding box
            min_r = min(r for r, c in pattern_cells)
            max_r = max(r for r, c in pattern_cells)
            min_c = min(c for r, c in pattern_cells)
            max_c = max(c for r, c in pattern_cells)

            # Extract pattern
            pattern = []
            for r in range(min_r, max_r + 1):
                row = []
                for c in range(min_c, max_c + 1):
                    row.append(grid[r][c])
                pattern.append(row)
            pattern_dict[color] = pattern

    def create_block(color, pattern):
        """Create 5x5 block with border and inverted pattern interior"""
        # Create 5x5 block with border of color
        block = [[color] * 5 for _ in range(5)]

        # Invert pattern and place in 3x3 interior
        # Pattern cells with color → background
        # Pattern cells with background → color
        for r in range(min(3, len(pattern))):
            for c in range(min(3, len(pattern[r]) if r < len(pattern) else 0)):
                if pattern[r][c] == color:
                    block[r + 1][c + 1] = background
                else:
                    block[r + 1][c + 1] = color

        return block

    # Determine grid arrangement of large blocks
    unique_rows = sorted(set(r for _, r, _ in large_blocks))
    unique_cols = sorted(set(c for _, _, c in large_blocks))

    row_map = {r: i for i, r in enumerate(unique_rows)}
    col_map = {c: i for i, c in enumerate(unique_cols)}

    # Create grid of blocks
    grid_rows = len(unique_rows)
    grid_cols = len(unique_cols)
    block_grid = [[None] * grid_cols for _ in range(grid_rows)]

    for color, r, c in large_blocks:
        gr = row_map[r]
        gc = col_map[c]
        pattern = pattern_dict.get(color, [[background] * 3 for _ in range(3)])
        block = create_block(color, pattern)
        block_grid[gr][gc] = block

    # Calculate output size with borders
    # Rows: top border (1) + blocks (5 each) + separators (1 between each) + bottom border (1)
    # = 1 + grid_rows * 5 + (grid_rows - 1) + 1 = grid_rows * 6 + 1
    output_rows = grid_rows * 6 + 1
    # Cols: left border (1) + blocks (5 each) + separators (1 between each) + right border (1)
    # = 1 + grid_cols * 5 + (grid_cols - 1) + 1 = grid_cols * 6 + 1
    output_cols = grid_cols * 6 + 1

    result = [[background] * output_cols for _ in range(output_rows)]

    # Place blocks in output
    for gr in range(grid_rows):
        for gc in range(grid_cols):
            if block_grid[gr][gc] is not None:
                block = block_grid[gr][gc]
                # Position: 1 for top border + gr*6 (5 for block + 1 for separator)
                start_r = 1 + gr * 6
                start_c = 1 + gc * 6

                for br in range(5):
                    for bc in range(5):
                        result[start_r + br][start_c + bc] = block[br][bc]

    return result
