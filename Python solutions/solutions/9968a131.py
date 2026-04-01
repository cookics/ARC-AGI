def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Even-indexed rows (0, 2, 4, ...) remain unchanged
    2. Odd-indexed rows (1, 3, 5, ...) undergo a transformation
    3. Background value is the most common element in the row
    4. Non-background elements shift right by 1 position
    5. Vacated positions are filled with the background value

    Procedure:
    1. Iterate through each row with its index
    2. If row index is even, keep the row unchanged
    3. If row index is odd, find the most common element (background)
    4. Shift all non-background elements right by 1 position
    5. Fill vacated positions with the background value
    """

    result = []

    for row_idx, row in enumerate(grid):
        if row_idx % 2 == 0:  # Even row - no change
            result.append(row[:])
        else:  # Odd row - apply transformation
            # Find most common element
            from collections import Counter

            counts = Counter(row)
            most_common_element = counts.most_common(1)[0][0]

            # Create new row starting with most common element
            new_row = [most_common_element] * len(row)

            # Find non-common elements and their positions
            non_common_positions = []
            for i, val in enumerate(row):
                if val != most_common_element:
                    non_common_positions.append((i, val))

            # Shift non-common elements right by 1 position
            for pos, val in non_common_positions:
                if pos + 1 < len(row):
                    new_row[pos + 1] = val

            result.append(new_row)

    return result
