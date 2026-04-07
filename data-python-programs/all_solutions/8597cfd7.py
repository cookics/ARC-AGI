def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided horizontally by a row of all 5s (separator)
    2. Values 2 and 4 appear above and below the separator
    3. Output is a 2x2 grid filled with either 2 or 4
    4. The fill value is determined by comparing counts in the section below the separator:
       - If count of 4s exceeds count of 2s by at least 2, output is all 4s
       - Otherwise, output is all 2s

    Procedure:
    1. Find the separator row (all 5s)
    2. Count occurrences of 2 and 4 in rows below the separator
    3. If (count_4s - count_2s >= 2), return 2x2 grid of 4s
    4. Otherwise, return 2x2 grid of 2s
    """

    # Find the row with line of 5s
    line_of_5s_row = -1
    for i, row in enumerate(grid):
        if all(cell == 5 for cell in row):
            line_of_5s_row = i
            break

    # Count 2s and 4s below the line of 5s
    count_2s = 0
    count_4s = 0

    for i in range(line_of_5s_row + 1, len(grid)):
        for cell in grid[i]:
            if cell == 2:
                count_2s += 1
            elif cell == 4:
                count_4s += 1

    # Determine output based on counts
    # 2 wins unless 4 has at least 2 more occurrences below the line
    if count_4s - count_2s >= 2:
        return [[4, 4], [4, 4]]
    else:
        return [[2, 2], [2, 2]]
