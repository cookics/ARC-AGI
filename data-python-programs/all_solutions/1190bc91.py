def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Vertical sequences spread their values horizontally in symmetric patterns
    2. Horizontal sequences spread their values vertically in symmetric patterns
    3. At each position in a sequence, spread other sequence values symmetrically based on distance
    4. Diagonal rays fill remaining cells

    Procedure:
    1. Find all sequences
    2. Create symmetric spreads from each sequence
    3. Shoot diagonal rays from sources (stopping at other sources)
    4. Apply priority rules for conflicts
    """

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Store all non-zero positions
    sources = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                sources.add((r, c))

    # Track sequence columns and rows for blocking diagonal rays
    seq_cols = {}  # col -> (start_row, end_row)
    seq_rows = {}  # row -> (start_col, end_col)

    # Store cell values with priority
    values = {}  # (r, c) -> list of (priority, value)

    # Find vertical sequences and create horizontal spreads
    for c in range(cols):
        r = 0
        while r < rows:
            if grid[r][c] != 0:
                start = r
                seq_vals = []
                while r < rows and grid[r][c] != 0:
                    seq_vals.append(grid[r][c])
                    r += 1
                # For each position in the vertical sequence, spread horizontally
                for idx, row in enumerate(range(start, start + len(seq_vals))):
                    # Spread sequence values symmetrically from this row
                    for dist in range(1, len(seq_vals)):
                        # Get value at distance 'dist' from current index (look backward first, then forward)
                        val = None
                        if idx - dist >= 0:
                            val = seq_vals[idx - dist]
                        elif idx + dist < len(seq_vals):
                            val = seq_vals[idx + dist]

                        if val is not None:
                            # Place symmetrically on both sides
                            for dc in [-dist, dist]:
                                nc = c + dc
                                if 0 <= nc < cols and (row, nc) not in sources:
                                    priority = (1, -c, row, dist)  # vertical spread
                                    if (row, nc) not in values:
                                        values[(row, nc)] = []
                                    values[(row, nc)].append((priority, val))
            else:
                r += 1

    # Find horizontal sequences and create vertical spreads
    for r in range(rows):
        c = 0
        while c < cols:
            if grid[r][c] != 0:
                start = c
                seq_vals = []
                while c < cols and grid[r][c] != 0:
                    seq_vals.append(grid[r][c])
                    c += 1
                # For each position in the horizontal sequence, spread vertically
                for idx, col in enumerate(range(start, start + len(seq_vals))):
                    # Spread sequence values symmetrically from this column
                    for dist in range(1, len(seq_vals)):
                        # Get value at distance 'dist' from current index (look backward first, then forward)
                        val = None
                        if idx - dist >= 0:
                            val = seq_vals[idx - dist]
                        elif idx + dist < len(seq_vals):
                            val = seq_vals[idx + dist]

                        if val is not None:
                            # Place symmetrically on both sides
                            for dr in [-dist, dist]:
                                nr = r + dr
                                if 0 <= nr < rows and (nr, col) not in sources:
                                    priority = (0, -col, r, dist)  # horizontal spread
                                    if (nr, col) not in values:
                                        values[(nr, col)] = []
                                    values[(nr, col)].append((priority, val))
            else:
                c += 1

    # Add diagonal rays from all sources
    for sr, sc in sources:
        val = grid[sr][sc]
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = sr + dr, sc + dc
            while 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) in sources:
                    break
                priority = (2, -sc, sr, abs(nr-sr))  # diagonal ray (lowest priority)
                if (nr, nc) not in values:
                    values[(nr, nc)] = []
                values[(nr, nc)].append((priority, val))
                nr += dr
                nc += dc

    # Build result
    result = [[0] * cols for _ in range(rows)]

    for (r, c), val_list in values.items():
        val_list.sort()
        result[r][c] = val_list[0][1]

    # Copy original source values
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                result[r][c] = grid[r][c]

    return result
