def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3x9 grid containing only 0s and 5s.
    2. Output is a 3x9 grid divided into three 3x3 blocks, each filled with a single value.
    3. The grid is processed in three vertical 3-column blocks (columns 0-2, 3-5, 6-8).
    4. For each 3-column block, we analyze which rows contain at least one 5 and count total 5s.
    5. The output value depends on the pattern of rows containing 5s and the total count.
    6. If only the top row (row 0) has 5s in a block, output value is 6.
    7. If only the middle row (row 1) has 5s in a block, output value is 4.
    8. If only the bottom row (row 2) has 5s in a block, output value is 1.
    9. If all three rows have 5s and total count is exactly 3 (one per row), output value is 9.
    10. If all three rows have 5s and total count is more than 3, output value is 3.

    Procedure:
    1. Create a 3x9 output grid initialized with zeros.
    2. Process each of the three 3-column blocks (0-2, 3-5, 6-8).
    3. For each block, identify which rows contain at least one 5 and count total 5s.
    4. Apply the mapping rules based on row pattern and count to determine output value.
    5. Fill the corresponding 3x3 block in the output grid with the determined value.
    6. Return the completed output grid.
    """

    # Create output grid
    result = [[0] * 9 for _ in range(3)]

    # Process each 3-column group
    for block_idx in range(3):
        start_col = block_idx * 3

        # Check which rows have 5s in this block and count total
        rows_with_5s = []
        total_5s = 0

        for row in range(3):
            has_5 = False
            for col in range(start_col, start_col + 3):
                if grid[row][col] == 5:
                    has_5 = True
                    total_5s += 1
            if has_5:
                rows_with_5s.append(row)

        # Map the pattern to output value
        if rows_with_5s == [0]:  # Only top row
            value = 6
        elif rows_with_5s == [1]:  # Only middle row
            value = 4
        elif rows_with_5s == [2]:  # Only bottom row
            value = 1
        elif len(rows_with_5s) == 3:  # All three rows
            if total_5s == 3:  # Exactly one 5 per row
                value = 9
            else:  # More than 3 total 5s
                value = 3
        else:  # 0 or 2 rows - shouldn't happen in examples
            value = 0

        # Fill this 3x3 block with the value
        for row in range(3):
            for col in range(start_col, start_col + 3):
                result[row][col] = value

    return result
