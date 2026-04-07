def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with various colored cells (non-zero values) and empty cells (0).
    2. For each row, identify the most frequent non-zero value.
    3. Transform each row to contain only that value and zeros, positioned according to specific rules.
    4. The count and positioning follow patterns based on input distribution.

    Procedure:
    1. For each row, count frequencies of all non-zero values.
    2. Find the most frequent non-zero value (break ties by first occurrence).
    3. Determine the positioning pattern and count based on input analysis.
    4. Apply transformation rules to place the value in the output.
    """

    result = []

    for row in grid:
        # Count frequencies of non-zero values
        freq = {}
        first_occurrence = {}

        for i, val in enumerate(row):
            if val != 0:
                if val not in freq:
                    freq[val] = 0
                    first_occurrence[val] = i
                freq[val] += 1

        if not freq:  # All zeros
            result.append([0] * len(row))
            continue

        # Find most frequent non-zero value (break ties by first occurrence)
        most_frequent = max(freq.keys(), key=lambda x: (freq[x], -first_occurrence[x]))

        # Find positions where the most frequent value appears
        positions = [i for i, val in enumerate(row) if val == most_frequent]
        input_count = len(positions)
        row_length = len(row)

        # Determine the transformation pattern
        new_row = [0] * row_length

        # Check for alternating patterns first
        if input_count == row_length // 2:
            if positions == list(range(1, row_length, 2)):
                # Odd positions → even positions
                for i in range(0, row_length, 2):
                    new_row[i] = most_frequent
                result.append(new_row)
                continue
            elif positions == list(range(0, row_length, 2)):
                # Even positions → odd positions
                for i in range(1, row_length, 2):
                    new_row[i] = most_frequent
                result.append(new_row)
                continue

        # Check if values form contiguous blocks
        contiguous_start = positions == list(range(input_count))
        contiguous_end = positions == list(range(row_length - input_count, row_length))

        # Output count is always the complement
        output_count = row_length - input_count

        # Determine position based on pattern
        if contiguous_start:
            # Values at start → move to end
            start_pos = row_length - output_count
            for i in range(start_pos, row_length):
                new_row[i] = most_frequent

        elif contiguous_end:
            # Values at end → move to start
            for i in range(output_count):
                new_row[i] = most_frequent

        else:
            # Scattered → place in middle
            start_pos = (row_length - output_count) // 2 + 1
            for i in range(start_pos, start_pos + output_count):
                new_row[i] = most_frequent

        result.append(new_row)

    return result
