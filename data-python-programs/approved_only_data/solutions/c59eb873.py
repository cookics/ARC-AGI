def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid of integers with dimensions m x n
    2. Output is a 2D grid of integers with dimensions 2m x 2n
    3. Each cell at position (i,j) in the input grid is expanded into a 2x2 block in the output grid
    4. The 2x2 block starts at position (2*i, 2*j) in the output grid and contains the same value as the input cell
    5. The transformation scales the grid by a factor of 2 in both dimensions

    Procedure:
    1. Create output grid with dimensions (2*input_height, 2*input_width)
    2. For each cell (i,j) in input, fill the 2x2 block at (2i,2j) to (2i+1,2j+1) with the same value
    """

    input_height = len(grid)
    input_width = len(grid[0])

    # Create output grid with double dimensions
    output = [[0 for _ in range(2 * input_width)] for _ in range(2 * input_height)]

    # Fill each 2x2 block
    for i in range(input_height):
        for j in range(input_width):
            value = grid[i][j]
            # Fill 2x2 block starting at (2*i, 2*j)
            output[2 * i][2 * j] = value
            output[2 * i][2 * j + 1] = value
            output[2 * i + 1][2 * j] = value
            output[2 * i + 1][2 * j + 1] = value

    return output
