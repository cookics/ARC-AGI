def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3x3 grid containing values 0 and 5.
    2. Output is a 6x6 grid containing values 0, 1, and 2.
    3. Each cell in the input grid maps to a 2x2 block in the output grid.
    4. If input cell contains 5, the corresponding 2x2 block becomes [[1,2], [2,1]].
    5. If input cell contains 0, the corresponding 2x2 block becomes [[0,0], [0,0]].
    6. The transformation scales the grid by 2x in both dimensions.

    Procedure:
    1. Initialize a 6x6 output grid filled with zeros.
    2. Iterate through each cell (i,j) in the input 3x3 grid.
    3. Calculate the corresponding 2x2 block position in output: (2*i:2*i+2, 2*j:2*j+2).
    4. If grid[i][j] == 5, place pattern [[1,2], [2,1]] in the 2x2 block.
    5. If grid[i][j] == 0, leave the 2x2 block as zeros (already initialized).
    6. Return the completed output grid.
    """

    rows, cols = len(grid), len(grid[0])
    output = [[0 for _ in range(cols * 2)] for _ in range(rows * 2)]

    for i in range(rows):
        for j in range(cols):
            # Calculate the top-left position of the 2x2 block in output
            out_i, out_j = i * 2, j * 2

            if grid[i][j] == 5:
                # Place the pattern [[1,2], [2,1]]
                output[out_i][out_j] = 1
                output[out_i][out_j + 1] = 2
                output[out_i + 1][out_j] = 2
                output[out_i + 1][out_j + 1] = 1
            else:  # grid[i][j] == 0
                # Place the pattern [[0,0], [0,0]] (already initialized to 0)
                pass

    return output
