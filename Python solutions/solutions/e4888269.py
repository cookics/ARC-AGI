def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has a separator dividing the grid (vertical column of 2s or horizontal row of 4s)
    2. One section contains ordered transformation pairs (e.g., 1→4, 6→7, 3→5, 4→6)
    3. Other section contains isolated values that need transformation
    4. Transformations follow transitive closure with order-aware stopping
    5. When following chain A→B→C→D, stop if next value appears as first element in an earlier pair

    Procedure:
    1. Detect separator (vertical column of 2s or horizontal row of 4s)
    2. Extract ordered transformation pairs from appropriate section
    3. For each isolated value, apply transitive closure with stopping rule
    4. Return transformed grid with pairs and separator unchanged
    """

    def follow_chain(value, ordered_pairs, transform_map):
        """Follow transformation chain with order-aware stopping rule"""
        current = value
        visited = set()

        while current in transform_map and current not in visited:
            visited.add(current)
            next_val = transform_map[current]

            # Check if next_val appears as first element in any earlier pair
            current_pair_index = -1
            for i, (first, second) in enumerate(ordered_pairs):
                if first == current:
                    current_pair_index = i
                    break

            # Look for next_val as first element in earlier pairs
            should_stop = False
            if current_pair_index != -1:
                for i, (first, second) in enumerate(ordered_pairs):
                    if i < current_pair_index and first == next_val:
                        should_stop = True
                        break

            # Move to next value
            current = next_val

            # If we should stop, break after moving
            if should_stop:
                break

        return current

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Check for vertical separator (column of 2s)
    separator_col = -1
    for col in range(cols):
        if all(grid[row][col] == 2 for row in range(rows)):
            separator_col = col
            break

    # Check for horizontal separator (row of 4s)
    separator_row = -1
    if separator_col == -1:
        for row in range(rows):
            if all(grid[row][col] == 4 for col in range(cols)):
                separator_row = row
                break

    # Extract ordered pairs
    ordered_pairs = []
    transform_map = {}

    if separator_col != -1:
        # Vertical separator format
        for row in range(rows):
            if grid[row][0] != 0 and grid[row][1] != 0:
                val1, val2 = grid[row][0], grid[row][1]
                ordered_pairs.append((val1, val2))
                transform_map[val1] = val2

        # Transform isolated values on the right side
        for row in range(rows):
            for col in range(separator_col + 1, cols):
                if result[row][col] != 0:
                    result[row][col] = follow_chain(
                        result[row][col], ordered_pairs, transform_map
                    )

    elif separator_row != -1:
        # Horizontal separator format
        for row in range(1, separator_row):
            if len(grid[row]) > 11 and grid[row][10] != 0 and grid[row][11] != 0:
                val1, val2 = grid[row][10], grid[row][11]
                ordered_pairs.append((val1, val2))
                transform_map[val1] = val2

        # Transform isolated values below separator
        for row in range(separator_row + 1, rows):
            for col in range(cols):
                if result[row][col] != 0:
                    result[row][col] = follow_chain(
                        result[row][col], ordered_pairs, transform_map
                    )

    return result
