def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an 11x11 grid divided into 3x3 blocks separated by rows/columns of 6s at indices 3 and 7
    2. Each 3x3 block can be: all 7s (empty), cross pattern with 2s, 5s, or 8s
    3. Cross pattern: [[7,X,7], [X,7,X], [7,X,7]] where X is 2, 5, or 8
    4. When only 2-crosses exist (plus empty blocks), we add one 5-cross and one 8-cross
    5. When 2-cross, 5-cross, and 8-cross all exist, output becomes 16x16 grid of all 7s
    6. New crosses are placed in positions that don't share rows/cols with existing 2-crosses

    Procedure:
    1. Parse the 11x11 grid into 9 blocks (3x3 arrangement)
    2. Identify the pattern type of each block (empty, 2-cross, 5-cross, 8-cross)
    3. Check if all three cross types exist - if yes, return 16x16 grid of 7s
    4. Otherwise, find positions where 2-crosses exist
    5. Place 5-cross and 8-cross in empty rows/columns based on strategy
    6. Reconstruct the 11x11 grid with the new patterns
    """

    # Parse the grid into 3x3 blocks
    blocks = {}
    for block_row in range(3):
        for block_col in range(3):
            block = []
            for r in range(3):
                row = []
                for c in range(3):
                    grid_r = block_row * 4 + r
                    grid_c = block_col * 4 + c
                    row.append(grid[grid_r][grid_c])
                block.append(row)
            blocks[(block_row, block_col)] = block

    # Identify pattern types
    def identify_pattern(block):
        if block == [[7, 7, 7], [7, 7, 7], [7, 7, 7]]:
            return "D"
        elif block == [[7, 2, 7], [2, 7, 2], [7, 2, 7]]:
            return "A"
        elif block == [[7, 5, 7], [5, 7, 5], [7, 5, 7]]:
            return "B"
        elif block == [[7, 8, 7], [8, 7, 8], [7, 8, 7]]:
            return "C"
        else:
            return "unknown"

    pattern_types = {}
    for pos, block in blocks.items():
        pattern_types[pos] = identify_pattern(block)

    # Check what patterns exist
    existing_patterns = set(pattern_types.values())

    if (
        "A" in existing_patterns
        and "B" in existing_patterns
        and "C" in existing_patterns
    ):
        # Output uniform 16x16 grid
        return [[7] * 16 for _ in range(16)]

    elif "A" in existing_patterns and existing_patterns.issubset({"A", "D"}):
        # Add B and C patterns
        result_blocks = blocks.copy()

        # Find A positions
        a_positions = [pos for pos, ptype in pattern_types.items() if ptype == "A"]

        # Find empty rows and columns
        occupied_rows = set(pos[0] for pos in a_positions)
        occupied_cols = set(pos[1] for pos in a_positions)
        empty_rows = {0, 1, 2} - occupied_rows
        empty_cols = {0, 1, 2} - occupied_cols

        # Determine placement strategy
        new_positions = []

        if len(empty_rows) == 1 and len(empty_cols) == 2:
            # Place both in the single empty row
            r = list(empty_rows)[0]
            cols = sorted(empty_cols)
            new_positions = [(r, cols[0]), (r, cols[1])]
        elif len(empty_rows) == 2 and len(empty_cols) == 1:
            # Place both in the single empty column
            c = list(empty_cols)[0]
            rows = sorted(empty_rows)
            new_positions = [(rows[0], c), (rows[1], c)]
        elif len(empty_rows) == 1 and len(empty_cols) == 1:
            # Place one in empty row (non-empty col), one in empty col (non-empty row)
            r = list(empty_rows)[0]
            c = list(empty_cols)[0]
            non_empty_cols = sorted(occupied_cols)
            non_empty_rows = sorted(occupied_rows)
            new_positions = [(r, non_empty_cols[0]), (non_empty_rows[-1], c)]

        # Assign 5 and 8 to positions
        # Sort positions lexicographically
        new_positions.sort()

        # Assign based on whether position is in empty row/col
        for pos in new_positions:
            r, c = pos
            in_empty_row = r in empty_rows
            in_empty_col = c in empty_cols

            if in_empty_row and not in_empty_col:
                # Empty row, non-empty col -> 5
                result_blocks[pos] = [[7, 5, 7], [5, 7, 5], [7, 5, 7]]
            elif not in_empty_row and in_empty_col:
                # Non-empty row, empty col -> 8
                result_blocks[pos] = [[7, 8, 7], [8, 7, 8], [7, 8, 7]]
            elif in_empty_row and in_empty_col:
                # Both empty -> assign based on position in sorted list
                if pos == new_positions[0]:
                    result_blocks[pos] = [[7, 5, 7], [5, 7, 5], [7, 5, 7]]
                else:
                    result_blocks[pos] = [[7, 8, 7], [8, 7, 8], [7, 8, 7]]

        # Reconstruct grid
        result = [[0] * 11 for _ in range(11)]

        # Fill separators
        for r in range(11):
            for c in range(11):
                if r == 3 or r == 7 or c == 3 or c == 7:
                    result[r][c] = 6

        # Fill blocks
        for block_row in range(3):
            for block_col in range(3):
                block = result_blocks[(block_row, block_col)]
                for r in range(3):
                    for c in range(3):
                        grid_r = block_row * 4 + r
                        grid_c = block_col * 4 + c
                        result[grid_r][grid_c] = block[r][c]

        return result

    else:
        # Unknown case, return input unchanged
        return grid
