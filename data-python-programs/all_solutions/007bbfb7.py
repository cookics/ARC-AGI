def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an n×n grid (in all examples, n=3)
    2. Output is an (n²)×(n²) grid (for n=3, that's 9×9)
    3. The output is conceptually divided into an n×n array of blocks, each block is n×n
    4. For position (i,j) in the input grid where grid[i][j] != 0, the block at position (i,j) contains a complete copy of the input
    5. For position (i,j) where grid[i][j] == 0, the block at position (i,j) is all zeros

    Procedure:
    1. Get the dimension n of the input grid
    2. Initialize an (n²)×(n²) output grid with zeros
    3. For each position (i,j) in the n×n input, if the value is non-zero, copy the entire input grid into the corresponding block starting at output row i*n, column j*n
    4. Return the output grid
    """
    n = len(grid)
    output_dim = n * n
    output = [[0] * output_dim for _ in range(output_dim)]

    # Iterate through each position in the input grid
    for block_row in range(n):
        for block_col in range(n):
            # Check if this position has a non-zero value
            if grid[block_row][block_col] != 0:
                # Calculate the starting position of this block in the output
                start_row = block_row * n
                start_col = block_col * n

                # Copy the entire input grid to this block
                for r in range(n):
                    for c in range(n):
                        output[start_row + r][start_col + c] = grid[r][c]

    result = output
    return result
