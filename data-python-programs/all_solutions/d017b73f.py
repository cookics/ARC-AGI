def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3-row grid
    2. Output is a 3-row grid with one row becoming all zeros
    3. The row that gets cleared is the one without a non-zero element in columns 0-1
    4. Elements from other rows get redistributed with specific patterns
    5. Empty columns are removed (compressed output)

    Procedure:
    1. Identify which row to clear (no non-zero in cols 0-1)
    2. Identify static region (cols 0 through max leftmost anchor of non-cleared rows)
    3. Extract chunks (contiguous non-zero segments) from each row
    4. Apply transformation:
       - Non-cleared rows: 1st chunk stays, other chunks swap to the other row
       - Cleared row chunks distribute to non-cleared rows
    5. Compress by removing empty columns
    """

    rows = len(grid)
    cols = len(grid[0])

    # Helper: find leftmost non-zero column in a row
    def leftmost_nonzero(row):
        for j, val in enumerate(row):
            if val != 0:
                return j
        return float('inf')

    # Helper: extract chunks (contiguous non-zero segments)
    def extract_chunks(row):
        chunks = []
        i = 0
        while i < len(row):
            if row[i] != 0:
                j = i
                while j < len(row) and row[j] != 0:
                    j += 1
                chunks.append((i, j, row[i:j]))
                i = j
            else:
                i += 1
        return chunks

    # Step 1: Identify which row to clear
    leftmost_cols = [leftmost_nonzero(grid[i]) for i in range(3)]
    cleared_row = None
    for i in range(3):
        if leftmost_cols[i] > 1:
            cleared_row = i
            break

    # If no row should be cleared, default to row 0
    if cleared_row is None:
        cleared_row = 0

    # Identify non-cleared rows
    active_rows = [i for i in range(3) if i != cleared_row]

    # Step 2: Find static region boundary
    static_bound = max(leftmost_cols[i] for i in active_rows if leftmost_cols[i] != float('inf'))

    # Step 3: Extract chunks from each row
    all_chunks = [extract_chunks(grid[i]) for i in range(3)]

    # Step 4: Create output grid (same size as input initially)
    output = [[0] * cols for _ in range(3)]

    # Place static region (unchanged)
    for i in active_rows:
        for j in range(min(static_bound + 1, cols)):
            output[i][j] = grid[i][j]

    # Place dynamic region chunks
    for i in active_rows:
        chunks = all_chunks[i]
        if not chunks:
            continue

        # First chunk (anchor) stays at original position
        start, end, values = chunks[0]
        for j, val in enumerate(values):
            if start + j < cols:
                output[i][start + j] = val

        # Other chunks behavior depends on which row is cleared
        other_row = active_rows[1] if i == active_rows[0] else active_rows[0]

        if cleared_row == 0:
            # Row 0 cleared: all non-anchor chunks go to other row
            for chunk_idx in range(1, len(chunks)):
                start, end, values = chunks[chunk_idx]
                for j, val in enumerate(values):
                    if start + j < cols:
                        output[other_row][start + j] = val
        else:
            # Row 2 cleared
            if i == 0:
                # Row 0: all chunks stay in place
                for chunk_idx in range(1, len(chunks)):
                    start, end, values = chunks[chunk_idx]
                    for j, val in enumerate(values):
                        if start + j < cols:
                            output[i][start + j] = val
            else:
                # Row 1: alternate - 2nd goes to other row, 3rd stays, 4th goes, etc.
                for chunk_idx in range(1, len(chunks)):
                    start, end, values = chunks[chunk_idx]
                    # Alternate: odd indices (1, 3, 5...) go to other row, even indices (2, 4...) stay
                    if chunk_idx % 2 == 1:
                        target = other_row
                    else:
                        target = i
                    for j, val in enumerate(values):
                        if start + j < cols:
                            output[target][start + j] = val

    # Place cleared row chunks
    cleared_chunks = all_chunks[cleared_row]
    if cleared_row == 0:
        # Row 0 cleared: if single chunk to row 1, if multiple chunks first to row 2, rest to row 1
        for chunk_idx, (start, end, values) in enumerate(cleared_chunks):
            if len(cleared_chunks) == 1:
                target_row = active_rows[0]  # row 1
            elif chunk_idx == 0:
                target_row = active_rows[1]  # row 2
            else:
                target_row = active_rows[0]  # row 1
            for j, val in enumerate(values):
                if start + j < cols:
                    output[target_row][start + j] = val
    else:
        # Row 2 cleared: all chunks to row 1
        target_row = 1 if 1 in active_rows else active_rows[0]
        for start, end, values in cleared_chunks:
            for j, val in enumerate(values):
                if start + j < cols:
                    output[target_row][start + j] = val

    # Step 5: Compress by removing empty columns
    non_empty_cols = []
    for j in range(cols):
        if any(output[i][j] != 0 for i in range(3)):
            non_empty_cols.append(j)

    result = []
    for i in range(3):
        result.append([output[i][j] for j in non_empty_cols])

    return result
