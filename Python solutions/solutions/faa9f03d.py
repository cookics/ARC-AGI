def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Final analysis of the pattern:
    1. Replace 2→3, 4→6
    2. Fill gaps in horizontal and vertical lines
    3. Extend vertical lines that have strong presence
    4. At intersections, vertical line value dominates

    Procedure:
    Apply transformations with correct vertical extension logic
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    # Step 1: Replace special numbers
    for i in range(rows):
        for j in range(cols):
            if result[i][j] == 2:
                result[i][j] = 3
            elif result[i][j] == 4:
                result[i][j] = 6

    # Step 2: Fill gaps in lines (iterative approach)
    for iteration in range(3):  # Multiple passes
        # Fill horizontal gaps
        for i in range(rows):
            for j in range(1, cols - 1):
                if result[i][j] == 0:
                    left = result[i][j - 1]
                    right = result[i][j + 1]
                    if left != 0 and left == right:
                        result[i][j] = left

        # Fill vertical gaps
        for j in range(cols):
            for i in range(1, rows - 1):
                if result[i][j] == 0:
                    up = result[i - 1][j]
                    down = result[i + 1][j]
                    if up != 0 and up == down:
                        result[i][j] = up

    # Step 3: Extend vertical lines based on strong patterns
    for j in range(cols):
        # Analyze the column
        non_zero_values = []
        for i in range(rows):
            if result[i][j] != 0:
                non_zero_values.append((i, result[i][j]))

        if len(non_zero_values) >= 3:
            # Find the most common value
            value_counts = {}
            for _, val in non_zero_values:
                value_counts[val] = value_counts.get(val, 0) + 1

            if value_counts:
                dominant_val = max(value_counts.keys(), key=lambda x: value_counts[x])
                dominant_count = value_counts[dominant_val]

                # If the dominant value appears frequently, extend it
                if dominant_count >= 3:
                    # Find the range where this value appears
                    positions = [i for i, val in non_zero_values if val == dominant_val]

                    if positions:
                        # Extend from the first occurrence to the end of significant presence
                        start_pos = min(positions)

                        # For values that span a significant range, extend them
                        if len(positions) >= 3:
                            # Extend downward from where the pattern starts
                            end_pos = max(positions)

                            # Check if we should extend further down
                            if end_pos < rows - 1:
                                # Look for more instances below
                                for check_i in range(end_pos + 1, rows):
                                    if result[check_i][j] == 0:
                                        # Count how many of the dominant value are above
                                        above_count = sum(
                                            1
                                            for k in range(check_i)
                                            if result[k][j] == dominant_val
                                        )
                                        if above_count >= 2:
                                            result[check_i][j] = dominant_val
                                        else:
                                            break

                            # Fill any gaps in the range
                            for fill_i in range(start_pos, end_pos + 1):
                                if result[fill_i][j] == 0:
                                    result[fill_i][j] = dominant_val

    # Step 4: Handle intersections where vertical lines cross horizontal lines
    for i in range(rows):
        for j in range(cols):
            current_val = result[i][j]
            if current_val != 0:
                # Check if there's a strong vertical line with different value
                column_values = {}
                for k in range(rows):
                    if result[k][j] != 0:
                        val = result[k][j]
                        column_values[val] = column_values.get(val, 0) + 1

                if len(column_values) > 1:
                    # Find the most common value in the column
                    dominant_vertical = max(
                        column_values.keys(), key=lambda x: column_values[x]
                    )

                    if (
                        dominant_vertical != current_val
                        and column_values[dominant_vertical] >= 3
                    ):
                        # Check if current position is part of a horizontal line
                        horizontal_count = 0
                        for k in range(cols):
                            if k != j and result[i][k] == current_val:
                                horizontal_count += 1

                        # If part of horizontal line, use vertical line value
                        if horizontal_count >= 2:
                            result[i][j] = dominant_vertical

    return result
