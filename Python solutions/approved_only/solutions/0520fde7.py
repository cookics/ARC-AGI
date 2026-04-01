def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3×7 grid where column 3 (0-indexed) contains all 5s acting as a vertical divider
    2. The divider separates the grid into two 3×3 sections: left (columns 0-2) and right (columns 4-6)
    3. Output is a 3×3 grid where each cell is 2 if both corresponding left and right cells are 1, else 0
    4. This is an AND operation between the two sections

    Procedure:
    1. For each row and column position in the 3×3 output
    2. Check if both left section (column c) and right section (column c+4) have value 1
    3. If both are 1, output 2; otherwise output 0
    """

    rows = len(grid)
    result = []

    for r in range(rows):
        row = []
        for c in range(3):  # Output is 3 columns
            left_val = grid[r][c]  # Left side: columns 0-2
            right_val = grid[r][c + 4]  # Right side: columns 4-6

            # AND operation: output 2 if both are 1, else 0
            if left_val == 1 and right_val == 1:
                row.append(2)
            else:
                row.append(0)

        result.append(row)

    return result
