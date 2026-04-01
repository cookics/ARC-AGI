def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided by a horizontal line of 7s into two parts.
    2. Top part contains pattern with 2s and 0s, bottom part contains pattern with 6s and 0s.
    3. Output grid has same dimensions as each pattern.
    4. Output contains 8 where both top and bottom patterns have 0 at the same position, otherwise 0.

    Procedure:
    1. Find the separator line of 7s
    2. Extract top and bottom patterns
    3. Compare each position: output 8 if both have 0, else 0
    """

    rows, cols = len(grid), len(grid[0])

    # Find the separator line of 7s
    sep_row = -1
    for r in range(rows):
        if all(grid[r][c] == 7 for c in range(cols)):
            sep_row = r
            break

    if sep_row == -1:
        return [[]]

    # Extract top and bottom patterns
    top_pattern = []
    bottom_pattern = []

    # Top pattern (before separator)
    for r in range(sep_row):
        top_pattern.append(grid[r])

    # Bottom pattern (after separator)
    for r in range(sep_row + 1, rows):
        bottom_pattern.append(grid[r])

    # Ensure both patterns have same dimensions
    if len(top_pattern) != len(bottom_pattern):
        return [[]]

    # Create output by comparing patterns
    result = []
    for r in range(len(top_pattern)):
        result_row = []
        for c in range(cols):
            # If both patterns have 0 at this position, output 8
            if top_pattern[r][c] == 0 and bottom_pattern[r][c] == 0:
                result_row.append(8)
            else:
                result_row.append(0)
        result.append(result_row)

    return result
