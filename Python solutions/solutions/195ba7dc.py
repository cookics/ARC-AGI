def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Each input row has exactly 13 elements with value 2 at index 6 acting as a separator.
    2. Left side contains 6 elements (indices 0-5) with values 0 or 7.
    3. Right side contains 6 elements (indices 7-12) with values 0 or 7.
    4. Output is a 6-element row performing logical OR between corresponding positions.
    5. Values 7 map to 1 in the output, values 0 map to 0 in the output.

    Procedure:
    1. For each input row, extract the left side (indices 0-5) and right side (indices 7-12).
    2. Skip the separator element at index 6.
    3. For each position i from 0 to 5, compare left_side[i] and right_side[i].
    4. If either element equals 7, append 1 to the output row, otherwise append 0.
    5. Return the resulting grid with all processed rows.
    """

    result = []
    for row in grid:
        # Extract left side (before separator) and right side (after separator)
        left_side = row[:6]
        right_side = row[7:]

        # Perform logical OR for each position
        output_row = []
        for i in range(6):
            # OR operation: 1 if either left[i] or right[i] is 7, else 0
            if left_side[i] == 7 or right_side[i] == 7:
                output_row.append(1)
            else:
                output_row.append(0)

        result.append(output_row)

    return result
