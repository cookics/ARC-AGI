def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 5×17 grid divided into three 5×5 sections separated by columns of 0s
    2. Section 1 (left): columns 0-4
    3. Section 2 (middle): columns 6-10
    4. Section 3 (right): columns 12-16
    5. Overlay rule: s2 > s3 > s1 priority where non-8 wins
    6. Special rule: if only ONE section has non-8 at a position, output 8 instead

    Procedure:
    1. Extract three sections
    2. For each position, check how many sections have non-8
    3. If only one has non-8, output 8
    4. Otherwise use priority overlay
    """

    # Extract the three 5×5 sections
    section1 = [[grid[i][j] for j in range(5)] for i in range(5)]
    section2 = [[grid[i][j] for j in range(6, 11)] for i in range(5)]
    section3 = [[grid[i][j] for j in range(12, 17)] for i in range(5)]

    result = []

    for i in range(5):
        row = []
        for j in range(5):
            val1 = section1[i][j]
            val2 = section2[i][j]
            val3 = section3[i][j]

            # Count how many sections have non-8 values
            non8_count = sum([val1 != 8, val2 != 8, val3 != 8])

            # If only one section has non-8, output 8
            if non8_count == 1:
                row.append(8)
            # Otherwise use priority: s2 > s3 > s1
            elif val2 != 8:
                row.append(val2)
            elif val3 != 8:
                row.append(val3)
            elif val1 != 8:
                row.append(val1)
            else:
                row.append(8)

        result.append(row)

    return result
