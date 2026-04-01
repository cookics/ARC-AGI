def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background 7s, marker 2s, and some 9s
    2. Output removes original 9s and fills between pairs of 2s with new 9s
    3. Only rows with exactly 2 occurrences of 2 are candidates for filling
    4. If N rows have exactly 2 twos, fill between 2s in first N-1 rows

    Procedure:
    1. Copy the input grid
    2. Remove all original 9s (set them to 7)
    3. Find rows with exactly 2 occurrences of 2
    4. Fill between 2s in the first N-1 such rows with 9
    """

    # Create a copy of the grid
    result = [row[:] for row in grid]

    # Remove all original 9s
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j] == 9:
                result[i][j] = 7

    # Find rows with exactly 2 occurrences of 2
    rows_with_two_twos = []
    for i in range(len(result)):
        twos = [j for j, val in enumerate(result[i]) if val == 2]
        if len(twos) == 2:
            rows_with_two_twos.append((i, twos[0], twos[1]))

    # Fill between 2s in the first N rows
    # Based on training cases: fill all except the last row that has exactly 2 twos
    fill_count = max(0, len(rows_with_two_twos) - 1)

    for idx in range(min(fill_count, len(rows_with_two_twos))):
        i, start, end = rows_with_two_twos[idx]
        for j in range(start + 1, end):
            result[i][j] = 9

    return result
