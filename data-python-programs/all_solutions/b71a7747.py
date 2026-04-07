def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided into regions by separator values
    2. Within the grid, there are 3x3 blocks at regular intervals containing "payload" values
    3. Payload values are the rarest values in the grid
    4. Output is formed by extracting 3x3 blocks at regular positions
    5. These blocks contain ONLY the payload values

    Procedure:
    1. Count frequency of all values in grid
    2. Try different combinations of the 2-3 rarest values as payload candidates
    3. For each payload set and each spacing/offset combination:
       - Extract blocks at regular positions
       - Check if all blocks contain ONLY payload values
       - If yes, return the extracted blocks
    """

    rows = len(grid)
    cols = len(grid[0])
    block_size = 3

    # Count value frequencies
    value_counts = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            value_counts[val] = value_counts.get(val, 0) + 1

    # Get all values sorted by frequency
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1])

    # Generate all possible 2-3 value combinations, sorted by total rarity
    candidate_payloads = []

    # Try 2-value combinations
    for i in range(len(sorted_values)):
        for j in range(i + 1, len(sorted_values)):
            v1, c1 = sorted_values[i]
            v2, c2 = sorted_values[j]
            candidate_payloads.append((c1 + c2, {v1, v2}))

    # Try 3-value combinations
    for i in range(len(sorted_values)):
        for j in range(i + 1, len(sorted_values)):
            for k in range(j + 1, len(sorted_values)):
                v1, c1 = sorted_values[i]
                v2, c2 = sorted_values[j]
                v3, c3 = sorted_values[k]
                candidate_payloads.append((c1 + c2 + c3, {v1, v2, v3}))

    # Sort by total count (rarest first)
    candidate_payloads.sort(key=lambda x: x[0])

    for _, payload_values in candidate_payloads:

        # Try different spacings and offsets
        for spacing in range(4, min(rows, cols)):
            for offset_r in range(min(spacing, rows)):
                for offset_c in range(min(spacing, cols)):
                    # Calculate block positions at regular intervals
                    row_positions = []
                    r = offset_r
                    while r + block_size <= rows:
                        row_positions.append(r)
                        r += spacing

                    col_positions = []
                    c = offset_c
                    while c + block_size <= cols:
                        col_positions.append(c)
                        c += spacing

                    # Need at least 3x3 grid of blocks
                    if len(row_positions) < 3 or len(col_positions) < 3:
                        continue

                    # Check if ALL blocks at these positions contain ONLY payload values
                    all_valid = True
                    for br in row_positions:
                        for bc in col_positions:
                            block_values = set()
                            for dr in range(block_size):
                                for dc in range(block_size):
                                    block_values.add(grid[br + dr][bc + dc])

                            if not block_values.issubset(payload_values):
                                all_valid = False
                                break
                        if not all_valid:
                            break

                    if all_valid:
                        # Found valid configuration - extract blocks
                        output_rows = len(row_positions) * block_size
                        output_cols = len(col_positions) * block_size
                        output = [[0] * output_cols for _ in range(output_rows)]

                        for i, br in enumerate(row_positions):
                            for j, bc in enumerate(col_positions):
                                for dr in range(block_size):
                                    for dc in range(block_size):
                                        output[i * block_size + dr][j * block_size + dc] = grid[br + dr][bc + dc]

                        return output

    # Fallback: return empty grid if no valid pattern found
    return [[]]
