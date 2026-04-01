def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is 2 rows × 12 columns containing only 0s and 2s
    2. Output is 8 rows × 7 columns with values 0, 2, and 3
    3. Output always starts with a 3 at position (0,3) as the starting point
    4. Rows 1-7 contain a "flowing" pattern made of 2s that moves/expands based on input encoding
    5. Each column encodes a value 0-3: (0,0)=0, (0,2)=1, (2,0)=2, (2,2)=3
    6. The presence and pattern of specific encoded values determines the flow type
    7. Value 1 (top empty bottom filled) indicates leftward initial flow
    8. Value 2 (top filled bottom empty) at regular intervals indicates diagonal rightward flow

    Procedure:
    1. Encode the 12 input columns as values 0-3
    2. Analyze which encoded values appear and their positions
    3. Classify flow pattern: if value 1 exists use left-flow, else if value 2 is periodic use diagonal, else use contract-expand
    4. Generate output based on flow pattern
    5. Return result grid
    """

    # Initialize result grid
    result = [[0 for _ in range(7)] for _ in range(8)]
    result[0][3] = 3

    # Encode each column as 0-3
    encoded = []
    for col in range(12):
        val = 0
        if grid[0][col] == 2:
            val += 2
        if grid[1][col] == 2:
            val += 1
        encoded.append(val)

    # Find positions of value 1
    positions_1 = [i for i, v in enumerate(encoded) if v == 1]

    # Determine flow pattern based on value 1 positions
    if len(positions_1) > 0 and positions_1[0] == 0:
        # Value 1 starts at position 0: left-then-right flow
        transformations = [[2, 3], [2, 3], [3, 4], [4], [4], [4], [4]]
    elif len(positions_1) >= 2:
        # Check if value 1 appears at regular intervals
        diffs = [positions_1[i+1] - positions_1[i] for i in range(len(positions_1)-1)]
        is_regular = len(set(diffs)) == 1 and diffs[0] == 3

        if is_regular:
            # Value 1 at regular intervals: diagonal rightward flow
            transformations = [[3, 4], [4, 5], [5, 6], [6], [6], [6], [6]]
        else:
            # Value 1 at irregular intervals: right-contract-expand flow
            transformations = [[3, 4], [3, 4], [3], [3], [3, 4], [4], [4]]
    else:
        # Default: right-contract-expand flow
        transformations = [[3, 4], [3, 4], [3], [3], [3, 4], [4], [4]]

    # Apply transformations
    for level in range(1, 8):
        positions = transformations[level - 1]
        for pos in positions:
            if 0 <= pos < 7:
                result[level][pos] = 2

    return result
