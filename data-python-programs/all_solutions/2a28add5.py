def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid where 7 is the background color and other values are markers
    2. Output is a grid where each row is either all 7s or has consecutive 8s in a 7 background
    3. For each row, count the distinct non-7 values
    4. If distinct count >= 3 AND value 6 is present, place total count of consecutive 8s
    5. The consecutive 8s are positioned to include the location of value 6
    6. Otherwise, output all 7s for that row

    Procedure:
    1. For each row, find all non-7 values and their positions
    2. Count distinct non-7 values and total non-7 values
    3. Check if distinct count >= 3 and value 6 is present
    4. If true, find position of value 6 and place total count of consecutive 8s centered around it
    5. If false, fill row with all 7s
    """

    result = []

    for row in grid:
        # Find non-7 positions and values
        non_seven_positions = []
        non_seven_values = []

        for i, val in enumerate(row):
            if val != 7:
                non_seven_positions.append(i)
                non_seven_values.append(val)

        # Count distinct non-7 values and total non-7 values
        distinct_values = set(non_seven_values)
        distinct_count = len(distinct_values)
        total_count = len(non_seven_values)

        # Create new row
        new_row = [7] * len(row)

        # Apply the rule
        if distinct_count >= 3 and 6 in distinct_values:
            # Need to place 'total_count' consecutive 8s
            # Key insight: place the 8s to include the position containing value 6
            best_start = 0

            # Find the position of value 6
            six_position = None
            for i, val in enumerate(row):
                if val == 6:
                    six_position = i
                    break

            if six_position is not None:
                # Place consecutive 8s to center the value 6 within them
                if total_count == 3:
                    # For 3-element sequences, prefer putting 6 in the middle
                    # but if 6 is near the end of the row, put 6 at end of sequence
                    if six_position >= len(row) - 2:  # 6 is in last 2 positions
                        ideal_start = six_position - 2  # Put 6 at end of sequence
                    else:
                        ideal_start = six_position - 1  # Put 6 in middle
                else:
                    ideal_start = six_position - (total_count + 1) // 2

                # Ensure the placement is within bounds
                best_start = max(0, min(ideal_start, len(row) - total_count))

                # Make sure the 6 is still included in the final placement
                if not (best_start <= six_position < best_start + total_count):
                    # If centering doesn't work, find the closest valid placement
                    if six_position < best_start:
                        best_start = six_position
                    else:
                        best_start = six_position - total_count + 1
                    # Ensure bounds again
                    best_start = max(0, min(best_start, len(row) - total_count))
            else:
                # Fallback if no 6 found (shouldn't happen based on our rule)
                best_start = 0

            # Place the consecutive 8s
            for i in range(best_start, best_start + total_count):
                new_row[i] = 8

        result.append(new_row)

    return result
