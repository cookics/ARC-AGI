def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3x3 grid with values 0 or 5
    2. Output is a 15x15 grid (5x scale factor)
    3. Each input cell with value 5 generates a 3x3 block of 5s at position (i*3+3, j*3+3)
    4. Off-diagonal cells (i≠j) with value 5 place the entire input pattern at (i*6, j*6)
    5. Diagonal corner cells (0,0) and (2,2) place input pattern at two symmetric positions
    6. Input rows that are entirely 5s make the corresponding output row (i*5+2) entirely 5s

    Procedure:
    1. Initialize 15x15 result grid with zeros
    2. For each cell (i,j) with value 5, place a 3x3 block at (i*3+3, j*3+3)
    3. For off-diagonal cells with value 5, place input pattern at (i*6, j*6)
    4. For diagonal corners (0,0) and (2,2), place input pattern at symmetric positions
    5. Check if any row is entirely 5s and fill corresponding output row
    """

    result = [[0] * 15 for _ in range(15)]

    # For each input cell with value 5
    for i in range(3):
        for j in range(3):
            if grid[i][j] == 5:
                # Rule 1: Always place 3x3 block at (i*3+3, j*3+3)
                for di in range(3):
                    for dj in range(3):
                        row, col = i * 3 + 3 + di, j * 3 + 3 + dj
                        if 0 <= row < 15 and 0 <= col < 15:
                            result[row][col] = 5

                # Rule 2 & 3: Place original pattern based on position
                patterns_to_place = []

                if i != j:  # Off-diagonal
                    patterns_to_place.append((i * 6, j * 6))
                else:  # Diagonal
                    if i == 0 and j == 0:
                        patterns_to_place.extend([(0, 3), (3, 0)])
                    elif i == 2 and j == 2:
                        patterns_to_place.extend([(9, 12), (12, 9)])
                    # (1,1) doesn't create original patterns, only 3x3 block

                # Place original patterns
                for start_row, start_col in patterns_to_place:
                    for di in range(3):
                        for dj in range(3):
                            row, col = start_row + di, start_col + dj
                            if 0 <= row < 15 and 0 <= col < 15:
                                result[row][col] = grid[di][dj]

    # Rule 4: Special row handling
    for i in range(3):
        if all(grid[i][j] == 5 for j in range(3)):
            output_row = i * 5 + 2
            for j in range(15):
                result[output_row][j] = 5

    return result
