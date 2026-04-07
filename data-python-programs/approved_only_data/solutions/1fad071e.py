def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 9x9 grid containing values 0, 1, and 2.
    2. Output is a single row with exactly 5 elements containing only 1s and 0s.
    3. The pattern is to count the number of 2x2 blocks of 1s in the input grid.
    4. The output contains that many 1s, followed by 0s to make the total length 5.

    Procedure:
    1. Scan the 9x9 grid to find all 2x2 blocks where all 4 cells contain 1.
    2. Count these blocks.
    3. Create output array with that many 1s, followed by 0s to total 5 elements.
    4. Return as a single-row 2D array.
    """

    rows, cols = len(grid), len(grid[0])
    count_1_blocks = 0

    # Scan for 2x2 blocks of 1s
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Check if 2x2 block starting at (i,j) contains all 1s
            if (
                grid[i][j] == 1
                and grid[i][j + 1] == 1
                and grid[i + 1][j] == 1
                and grid[i + 1][j + 1] == 1
            ):
                count_1_blocks += 1

    # Create output array: count_1_blocks ones followed by zeros to total 5
    result = [1] * count_1_blocks + [0] * (5 - count_1_blocks)

    return [result]
