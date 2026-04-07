def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 3×3 blocks of non-zero values separated by borders (0s) and separator lines
    2. Separator lines are rows or columns filled with a single non-zero value
    3. Major separators (values 5, 7, 9) divide the grid into independent regions
    4. Minor separators (value 4) are used for transformations within regions
    5. Output transforms blocks within each region based on these rules:
       - If N-1 blocks are value 1 and one is different, all blocks become that different value
       - Else if separator value appears as a block value, all blocks become separator value
       - Else blocks swap across the separator (horizontal swap for vertical separator, vertical swap for horizontal separator)

    Procedure:
    1. Find all 3×3 uniform blocks in the grid
    2. Find separator lines and distinguish major from minor separators
    3. Partition grid by major separators into independent regions
    4. For each region, find blocks and minor separators
    5. Apply transformation rules and update result grid
    """

    def fill_block(result, r, c, val):
        """Fill a 3×3 block starting at (r, c) with value val."""
        for dr in range(3):
            for dc in range(3):
                result[r + dr][c + dc] = val

    def apply_transformation(result, blocks, h_sep, v_sep):
        """Apply transformation rules to a group of blocks."""
        if len(blocks) == 0:
            return

        block_values = [val for r, c, val in blocks]

        # Rule 1: If N-1 blocks are value 1, replace all with the minority value
        count_ones = sum(1 for val in block_values if val == 1)
        if count_ones == len(blocks) - 1:
            minority_val = next(val for val in block_values if val != 1)
            for r, c, _ in blocks:
                fill_block(result, r, c, minority_val)
            return

        # Rule 2: If separator value appears as a block value, replace all with separator value
        sep_val = None
        if h_sep:
            sep_val = h_sep[1]
        elif v_sep:
            sep_val = v_sep[1]

        if sep_val and sep_val in block_values:
            for r, c, _ in blocks:
                fill_block(result, r, c, sep_val)
            return

        # Rule 3: Swap blocks across separator
        if v_sep:
            # Vertical separator: swap horizontally
            sep_col = v_sep[0]
            left_blocks = [(r, c, val) for r, c, val in blocks if c < sep_col]
            right_blocks = [(r, c, val) for r, c, val in blocks if c > sep_col]

            # Group by row
            left_by_row = {}
            for r, c, val in left_blocks:
                left_by_row[r] = (r, c, val)

            right_by_row = {}
            for r, c, val in right_blocks:
                right_by_row[r] = (r, c, val)

            # Swap
            for row in left_by_row:
                if row in right_by_row:
                    r1, c1, val1 = left_by_row[row]
                    r2, c2, val2 = right_by_row[row]
                    fill_block(result, r1, c1, val2)
                    fill_block(result, r2, c2, val1)

        elif h_sep:
            # Horizontal separator: swap vertically
            sep_row = h_sep[0]
            top_blocks = [(r, c, val) for r, c, val in blocks if r < sep_row]
            bottom_blocks = [(r, c, val) for r, c, val in blocks if r > sep_row]

            # Group by column
            top_by_col = {}
            for r, c, val in top_blocks:
                top_by_col[c] = (r, c, val)

            bottom_by_col = {}
            for r, c, val in bottom_blocks:
                bottom_by_col[c] = (r, c, val)

            # Swap
            for col in top_by_col:
                if col in bottom_by_col:
                    r1, c1, val1 = top_by_col[col]
                    r2, c2, val2 = bottom_by_col[col]
                    fill_block(result, r1, c1, val2)
                    fill_block(result, r2, c2, val1)

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find all 3×3 uniform blocks
    blocks = []
    for r in range(rows - 2):
        for c in range(cols - 2):
            val = grid[r][c]
            if val == 0:
                continue

            # Check if this 3×3 region is uniform
            is_uniform = all(
                grid[r + dr][c + dc] == val
                for dr in range(3)
                for dc in range(3)
            )

            if is_uniform:
                blocks.append((r, c, val))

    # Remove overlapping blocks (keep top-left corners only)
    unique_blocks = []
    seen = set()
    for r, c, val in blocks:
        if (r, c) not in seen:
            seen.add((r, c))
            # Mark all cells of this block
            for dr in range(3):
                for dc in range(3):
                    seen.add((r + dr, c + dc))
            unique_blocks.append((r, c, val))

    blocks = unique_blocks

    # Find separator lines
    h_separators = []  # (row, value)
    for r in range(rows):
        if grid[r][0] == 0:
            continue
        val = grid[r][0]
        if all(grid[r][c] == val for c in range(cols)):
            h_separators.append((r, val))

    v_separators = []  # (col, value)
    for c in range(cols):
        if grid[0][c] == 0:
            continue
        val = grid[0][c]
        if all(grid[r][c] == val for r in range(rows)):
            v_separators.append((c, val))

    # Process blocks by grouping them into sections
    # Distinguish between major separators (5, 7, 9, etc.) and minor separators (4)
    # Major separators divide the grid into independent regions
    # Minor separators are used for transformations within regions

    major_h_seps = [r for r, v in h_separators if v != 4]
    major_v_seps = [c for c, v in v_separators if v != 4]

    # Create boundaries
    h_boundaries = sorted(set([0] + major_h_seps + [rows]))
    v_boundaries = sorted(set([0] + major_v_seps + [cols]))

    # Process each section
    for i in range(len(h_boundaries) - 1):
        for j in range(len(v_boundaries) - 1):
            r1, r2 = h_boundaries[i], h_boundaries[i + 1]
            c1, c2 = v_boundaries[j], v_boundaries[j + 1]

            # Skip separator rows/cols
            if r2 - r1 <= 1 or c2 - c1 <= 1:
                continue

            # Find blocks in this section
            section_blocks = [
                (r, c, val) for r, c, val in blocks
                if r1 <= r < r2 and c1 <= c < c2
            ]

            if len(section_blocks) == 0:
                continue

            # Find separator within this section
            local_h_sep = None
            local_v_sep = None

            for r in range(r1 + 1, r2 - 1):
                if grid[r][c1] == 0:
                    continue
                val = grid[r][c1]
                if all(grid[r][c] == val for c in range(c1, c2) if c < cols):
                    local_h_sep = (r, val)
                    break

            for c in range(c1 + 1, c2 - 1):
                if grid[r1][c] == 0:
                    continue
                val = grid[r1][c]
                if all(grid[r][c] == val for r in range(r1, r2) if r < rows):
                    local_v_sep = (c, val)
                    break

            # Apply transformation
            apply_transformation(result, section_blocks, local_h_sep, local_v_sep)

    return result
