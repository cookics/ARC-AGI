def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. 3x3 frames tile horizontally with period 3 aligned to their center column
    2. Frames with same border color share a common left boundary (min of all their left edges)
    3. Each frame extends right to match blocks with border/center colors
    4. Left extension: if no same-border frames to left, extend to leftmost matching block
    5. Preserve columns left of shared boundary only if they're part of large blocks

    Procedure:
    1. Find all 3x3 frames
    2. Find all large blocks (connected components > 9 cells)
    3. Group frames by border color
    4. For each frame, compute tiling range and fill with pattern
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find frames
    def is_frame(r, c):
        if r < 1 or r >= rows - 1 or c < 1 or c >= cols - 1:
            return False
        center = grid[r][c]
        border = grid[r-1][c-1]
        if center == border or center == 8 or border == 8:
            return False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if grid[r+dr][c+dc] != border:
                    return False
        return True

    patterns = []
    for r in range(rows):
        for c in range(cols):
            if is_frame(r, c):
                patterns.append((r, c, grid[r-1][c-1], grid[r][c]))

    # Find large blocks
    visited = set()
    blocks = []
    for r in range(rows):
        for c in range(cols):
            if (r,c) in visited or grid[r][c] == 8:
                continue
            color = grid[r][c]
            q = [(r,c)]
            cells = []
            while q:
                cr, cc = q.pop(0)
                if (cr,cc) in visited or cr < 0 or cr >= rows or cc < 0 or cc >= cols or grid[cr][cc] != color:
                    continue
                visited.add((cr,cc))
                cells.append((cr,cc))
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    q.append((cr+dr,cc+dc))
            if len(cells) > 9:
                minr = min(r for r,c in cells)
                maxr = max(r for r,c in cells)
                minc = min(c for r,c in cells)
                maxc = max(c for r,c in cells)
                blocks.append((minr, maxr, minc, maxc, color, set(cells)))

    # Group by border
    border_groups = {}
    for pr, pc, border, center in patterns:
        if border not in border_groups:
            border_groups[border] = []
        border_groups[border].append((pr, pc, center))

    # Process each frame
    for pr, pc, border, center in patterns:
        # Find shared left boundary for this border color
        same_border = border_groups[border]
        shared_left = min(c - 1 for r, c, ctr in same_border)

        # Check if there are frames with same border to the left
        frames_to_left = [c for r, c, ctr in same_border if c < pc]

        # Determine left boundary
        if not frames_to_left:
            # No frames to left - extend to leftmost matching block
            left = shared_left
            for minr, maxr, minc, maxc, color, cells in blocks:
                if maxr >= pr - 1 and minr <= pr + 1 and color == center:
                    left = min(left, minc)
        else:
            # Has frames to left - use shared boundary but preserve blocks' leftmost columns
            left = shared_left

        # Find right boundary
        right = pc + 1
        for minr, maxr, minc, maxc, color, cells in blocks:
            if maxr >= pr - 1 and minr <= pr + 1:
                if color == border or color == center:
                    right = max(right, maxc)

        # Tile the pattern
        for row_offset in [-1, 0, 1]:
            r = pr + row_offset
            for c in range(left, right + 1):
                # Check if we should skip (preserve) this cell
                skip = False
                for minr, maxr, minc, maxc, color, cells in blocks:
                    if (r, c) in cells:
                        # Skip cells in blocks with non-border/non-center color
                        if color != border and color != center:
                            skip = True
                            break
                        # Preserve leftmost column of center-color blocks left of shared boundary
                        if color == center and c == minc and c < shared_left:
                            skip = True
                            break

                if not skip:
                    if row_offset == 0:
                        result[r][c] = center if (c - pc) % 3 == 0 else border
                    else:
                        result[r][c] = border

    return result
