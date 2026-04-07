def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Each row in the output is either all 0s or all 5s.
    2. If all elements in an input row are the same value, the output row becomes [5,5,5].
    3. If the elements in an input row are not all the same, the output row becomes [0,0,0].

    Procedure:
    1. For each row in the input grid:
    2. Check if all elements in the row are identical
    3. If yes, set the corresponding output row to [5,5,5]
    4. If no, set the corresponding output row to [0,0,0]
    """

    result = []

    for row in grid:
        # Check if all elements in the row are the same
        if all(element == row[0] for element in row):
            # All elements are the same, output [5,5,5]
            result.append([5] * len(row))
        else:
            # Not all elements are the same, output [0,0,0]
            result.append([0] * len(row))

    return result
