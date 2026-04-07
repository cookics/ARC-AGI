def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid filled mostly with 3s, with one row containing a pattern of 7s and 3s
    2. Output generates additional rows below the pattern row with alternating 6s and 7s
    3. Each generated row is placed 2 rows after the previous generated row
    4. Values decrease in count following specific filtering rules:
       - First: 6s at all positions where pattern row had 3s
       - Second: 7s at isolated (not in runs), non-edge positions where pattern had 7s
       - Subsequent: extract middle elements from consecutive runs (spacing 2) of previous positions
    5. Generation stops when no more positions qualify

    Procedure:
    1. Find the row containing the 7s pattern
    2. Extract positions of 3s and 7s from that row
    3. Generate sequences of positions alternating between 6s and 7s
    4. Place them in output at intervals of 2 rows from the pattern row
    """

    result = [row[:] for row in grid]
    height = len(grid)
    width = len(grid[0])

    # Find the pattern row (contains 7s)
    pattern_row_idx = -1
    for i in range(height):
        if 7 in grid[i]:
            pattern_row_idx = i
            break

    if pattern_row_idx == -1:
        return result

    pattern_row = grid[pattern_row_idx]

    # Extract positions of 3s and 7s
    pos_3 = [i for i in range(width) if pattern_row[i] == 3]
    pos_7 = [i for i in range(width) if pattern_row[i] == 7]

    # Helper: Identify padding (runs of 3+ same values at edges)
    def find_padding():
        """Find padding regions: runs of 3+ consecutive same values at edges"""
        padding_positions = set()

        # Check start padding
        if len(pattern_row) >= 3:
            start_val = pattern_row[0]
            count = 1
            for i in range(1, len(pattern_row)):
                if pattern_row[i] == start_val:
                    count += 1
                else:
                    break
            if count >= 3:
                for i in range(count):
                    padding_positions.add(i)

        # Check end padding
        if len(pattern_row) >= 3:
            end_val = pattern_row[-1]
            count = 1
            for i in range(len(pattern_row) - 2, -1, -1):
                if pattern_row[i] == end_val:
                    count += 1
                else:
                    break
            if count >= 3:
                for i in range(len(pattern_row) - count, len(pattern_row)):
                    padding_positions.add(i)

        return padding_positions

    padding = find_padding()

    # Helper: Check if position is part of a consecutive run in input
    def is_in_run(pos, all_positions):
        """Check if a position is part of a run of 2+ consecutive positions"""
        in_run = False
        if pos > 0 and (pos - 1) in all_positions:
            in_run = True
        if pos < width - 1 and (pos + 1) in all_positions:
            in_run = True
        return in_run

    # Helper: Get isolated 3 positions (not in runs, not in padding)
    def get_isolated_3s(positions_3):
        """Get 3 positions that are isolated (not in runs) and not in padding"""
        set_3 = set(positions_3)
        result = []
        for pos in positions_3:
            # Check if in padding
            if pos in padding:
                continue
            # Check if in run
            if is_in_run(pos, set_3):
                continue
            result.append(pos)
        return result

    # Helper: Get isolated non-edge 7 positions
    def get_isolated_noedge_7s(positions_7):
        """Get 7 positions that are isolated (not in runs), not at edges, not in/adjacent to padding"""
        set_7 = set(positions_7)
        result = []
        for pos in positions_7:
            # Check if edge
            if pos == 0 or pos == width - 1:
                continue
            # Check if in padding or adjacent to padding
            if pos in padding:
                continue
            if pos > 0 and (pos - 1) in padding:
                continue
            if pos < width - 1 and (pos + 1) in padding:
                continue
            # Check if in run
            if is_in_run(pos, set_7):
                continue
            result.append(pos)
        return result

    # Helper: Group positions into runs (consecutive with spacing 2)
    def group_into_runs(positions):
        """Group positions into runs where consecutive positions differ by 2"""
        if not positions:
            return []

        runs = []
        current_run = [positions[0]]

        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] == 2:
                current_run.append(positions[i])
            else:
                runs.append(current_run)
                current_run = [positions[i]]
        runs.append(current_run)

        return runs

    # Helper: Get middle elements from runs
    def get_middle_elements(positions):
        """Get middle elements from runs of positions"""
        runs = group_into_runs(positions)
        result = []

        for run in runs:
            length = len(run)
            if length >= 3:
                # For odd length: select middle element
                # For even length: select two middle elements
                if length % 2 == 1:
                    result.append(run[length // 2])
                else:
                    result.append(run[length // 2 - 1])
                    result.append(run[length // 2])

        return result

    # Generate sequences
    sequences = []

    # Round 1: isolated 3 positions → 6s
    isolated_3s = get_isolated_3s(pos_3)
    if isolated_3s:
        sequences.append((6, isolated_3s))

    # Round 2: isolated non-edge 7 positions → 7s
    isolated_7s = get_isolated_noedge_7s(pos_7)
    if isolated_7s:
        sequences.append((7, isolated_7s))

    # Subsequent rounds: middle elements from previous rounds
    prev_6_positions = isolated_3s
    prev_7_positions = isolated_7s

    while True:
        # Try to generate next 6s row
        next_6_positions = get_middle_elements(prev_6_positions)
        if next_6_positions:
            sequences.append((6, next_6_positions))
            prev_6_positions = next_6_positions
        else:
            break

        # Try to generate next 7s row
        next_7_positions = get_middle_elements(prev_7_positions)
        if next_7_positions:
            sequences.append((7, next_7_positions))
            prev_7_positions = next_7_positions
        else:
            break

    # Place sequences in output
    current_row = pattern_row_idx + 2
    for value, positions in sequences:
        if current_row >= height:
            break
        for pos in positions:
            result[current_row][pos] = value
        current_row += 2

    return result
