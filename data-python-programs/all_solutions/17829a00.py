def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has two border rows: row 0 (top_color) and row 15 (bottom_color)
    2. Middle rows contain scattered instances of these colors plus background (7)
    3. Rows with each color are grouped into contiguous blocks
    4. Blocks are merged: if 2+ blocks, merge first and last; if 1 block, split and merge halves
    5. Merged blocks are placed adjacent to their respective borders

    Procedure:
    1. Identify rows containing top_color and bottom_color
    2. Group into contiguous blocks
    3. Merge blocks and place near borders
    4. Handle vertical column extensions (3+ consecutive cells)
    """

    H, W = len(grid), len(grid[0])
    result = [[7] * W for _ in range(H)]
    result[0] = grid[0][:]
    result[-1] = grid[-1][:]

    top_color = grid[0][0]
    bottom_color = grid[-1][0]

    # Find rows containing each color
    top_rows = [r for r in range(1, H-1) if any(grid[r][c] == top_color for c in range(W))]
    bottom_rows = [r for r in range(1, H-1) if any(grid[r][c] == bottom_color for c in range(W))]

    # Find contiguous blocks
    def get_blocks(rows):
        if not rows:
            return []
        blocks, curr = [], [rows[0]]
        for i in range(1, len(rows)):
            if rows[i] == rows[i-1] + 1:
                curr.append(rows[i])
            else:
                blocks.append(curr)
                curr = [rows[i]]
        blocks.append(curr)
        return blocks

    top_blocks = get_blocks(top_rows)
    bottom_blocks = get_blocks(bottom_rows)

    # Process top color
    if len(top_blocks) >= 2:
        b1, b2 = top_blocks[0], top_blocks[-1]
        out_size = max(len(b1), len(b2))
        for i in range(out_size):
            for c in range(W):
                if (i < len(b1) and grid[b1[i]][c] == top_color) or \
                   (i < len(b2) and grid[b2[i]][c] == top_color):
                    result[i + 1][c] = top_color

        # Extend vertical segments
        for c in range(W):
            # Check b2 for 3+ consecutive
            max_consecutive_in_b2 = 0
            consecutive = 0
            for r in b2:
                if grid[r][c] == top_color:
                    consecutive += 1
                    max_consecutive_in_b2 = max(max_consecutive_in_b2, consecutive)
                else:
                    consecutive = 0

            if max_consecutive_in_b2 >= 3:
                # Fill entire output range
                for i in range(1, out_size + 1):
                    result[i][c] = top_color
            elif len(b1) >= 2 and len(b2) == 1:
                # Check if last 2 rows of b1 have the value
                if grid[b1[-1]][c] == top_color and grid[b1[-2]][c] == top_color:
                    # Fill only first 2 rows
                    for i in range(1, 3):
                        result[i][c] = top_color
    elif len(top_blocks) == 1:
        for i, r in enumerate(top_blocks[0]):
            for c in range(W):
                if grid[r][c] == top_color:
                    result[i + 1][c] = top_color

    # Process bottom color
    if len(bottom_blocks) == 1:
        block = bottom_blocks[0]
        n = len(block)
        mid = (n + 1) // 2
        b1, b2 = block[:mid], block[mid:]

        out_size = max(len(b1), len(b2))
        start = H - 1 - out_size

        # Copy first half to output
        for i in range(out_size):
            if i < len(b1):
                for c in range(W):
                    if grid[b1[i]][c] == bottom_color:
                        result[start + i][c] = bottom_color

        # OR last row of second half into last output row
        if len(b2) > 0:
            for c in range(W):
                if grid[b2[-1]][c] == bottom_color:
                    result[start + out_size - 1][c] = bottom_color

        # Extend vertical segments (3+ consecutive cells)
        for c in range(W):
            max_consecutive = 0
            consecutive = 0
            for r in block:
                if grid[r][c] == bottom_color:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0

            if max_consecutive >= 3:
                for i in range(start - 1, H - 1):
                    result[i][c] = bottom_color

    elif len(bottom_blocks) >= 2:
        b1, b2 = bottom_blocks[0], bottom_blocks[-1]
        out_size = max(len(b1), len(b2))
        start = H - 1 - out_size

        for i in range(out_size):
            for c in range(W):
                if (i < len(b1) and grid[b1[i]][c] == bottom_color) or \
                   (i < len(b2) and grid[b2[i]][c] == bottom_color):
                    result[start + i][c] = bottom_color

    return result
