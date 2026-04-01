def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains square NxN rectangles of colors (ignoring background/noise)
    2. Output is always 5 rows tall showing a "side view" of stacked rectangles
    3. Each block shows rectangles stacked by size (smallest on top)
    4. Within each block:
       - Row i shows colors of rectangles with height > i
       - Smallest rectangle takes its full width, others take 1 column each
    5. Blocks are distributed greedily and separated by column of 1s
    6. Block width = max_size_in_block + 1

    Procedure:
    1. Find all square colored rectangles
    2. Count by (color, size) and determine background
    3. Create blocks greedily, taking one from each group per block
    4. Render each block as a cross-section view
    """
    from collections import Counter, defaultdict

    def find_rectangles(grid):
        """Find all square colored rectangles by scanning and marking."""
        rows, cols = len(grid), len(grid[0])

        # Determine which values to skip
        all_vals = [grid[i][j] for i in range(rows) for j in range(cols)]
        val_counts = Counter(all_vals)

        # Most common is background
        background = val_counts.most_common(1)[0][0]

        # Also skip 1 and 8 if they appear
        skip_set = {background}
        if 1 in val_counts:
            skip_set.add(1)
        if 8 in val_counts:
            skip_set.add(8)

        visited = [[False] * cols for _ in range(rows)]
        rectangles = []

        # Scan from largest to smallest size
        for size in range(min(rows, cols), 1, -1):
            for r in range(rows - size + 1):
                for c in range(cols - size + 1):
                    # Skip if any cell already used
                    overlap = False
                    for dr in range(size):
                        for dc in range(size):
                            if visited[r+dr][c+dc]:
                                overlap = True
                                break
                        if overlap:
                            break
                    if overlap:
                        continue

                    # Skip background/noise cells
                    if grid[r][c] in skip_set:
                        continue

                    color = grid[r][c]

                    # Verify all cells match the color AND none are background
                    is_valid_rect = True
                    for dr in range(size):
                        for dc in range(size):
                            cell_val = grid[r+dr][c+dc]
                            if cell_val != color or cell_val in skip_set:
                                is_valid_rect = False
                                break
                        if not is_valid_rect:
                            break

                    if is_valid_rect:
                        # Mark all cells as visited
                        for dr in range(size):
                            for dc in range(size):
                                visited[r+dr][c+dc] = True
                        rectangles.append((color, size))

        return rectangles

    rectangles = find_rectangles(grid)

    # Group and count by (color, size)
    group_counts = defaultdict(int)
    for color, size in rectangles:
        group_counts[(color, size)] += 1

    # Determine max number of blocks to create
    # Sort shapes by size (descending) and get counts
    sorted_by_size = sorted(group_counts.items(), key=lambda x: x[0][1], reverse=True)
    counts_by_size_desc = [count for (color, size), count in sorted_by_size]

    if len(counts_by_size_desc) >= 2:
        largest_count = counts_by_size_desc[0]
        second_largest_count = counts_by_size_desc[1]

        # Rule: if largest > second_largest, create largest blocks
        #       else create largest + min(largest, second_largest) blocks
        if largest_count > second_largest_count:
            max_blocks = largest_count
        else:
            max_blocks = largest_count + min(largest_count, second_largest_count)
    else:
        # Only one shape type
        max_blocks = counts_by_size_desc[0] if counts_by_size_desc else 0

    # Create blocks greedily
    blocks = []
    remaining = dict(group_counts)

    while any(count > 0 for count in remaining.values()) and len(blocks) < max_blocks:
        block = []
        for key in sorted(remaining.keys(), key=lambda x: x[1]):  # By size
            if remaining[key] > 0:
                block.append(key)
                remaining[key] -= 1
        if block:
            blocks.append(block)

    # Render output
    result = [[] for _ in range(5)]

    for block in blocks:
        block = sorted(block, key=lambda x: x[1])  # Smallest first
        max_size = max(size for _, size in block)

        for row in range(5):
            # Which rectangles are visible at this height?
            visible = [(color, size) for color, size in block if row < size]

            if visible:
                # First (smallest) gets full width
                first_color, first_size = visible[0]
                block_row = [first_color] * first_size

                # Rest get 1 column each
                for color, size in visible[1:]:
                    block_row.append(color)

                # Pad to max_size
                block_row.extend([1] * (max_size - len(block_row)))
            else:
                block_row = [1] * max_size

            result[row].extend(block_row)
            result[row].append(1)  # Separator

    return result
