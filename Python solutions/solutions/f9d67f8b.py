def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with some cells containing value 9 (corrupted cells)
    2. Output is the same grid with 9s replaced by their actual values
    3. The grid has both vertical (row-wise) and horizontal (column-wise) reflection symmetry
    4. Cell (i, j) with value 9 is first restored using (31-i, j) - same column, symmetric row
    5. If that also has 9, fallback to (i, 31-j) - same row, symmetric column
    6. The conceptual grid is 32x32, with symmetry about row 15.5 and column 15.5

    Procedure:
    1. For each cell (i, j):
       - If contains 9, try row-wise symmetry first: (31-i, j)
       - If that's also 9, try column-wise symmetry: (i, 31-j)
       - Otherwise, keep original value
    2. Return the restored grid
    """
    n = len(grid)
    result = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if grid[i][j] == 9:
                sym_i = n + 1 - i
                if sym_i < n and grid[sym_i][j] != 9:
                    result[i][j] = grid[sym_i][j]
                else:
                    sym_j = n + 1 - j
                    if sym_j < n and grid[i][sym_j] != 9:
                        result[i][j] = grid[i][sym_j]
                    else:
                        result[i][j] = grid[i][j]
            else:
                result[i][j] = grid[i][j]

    return result
