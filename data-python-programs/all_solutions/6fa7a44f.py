def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid (list of lists) containing integers.
    2. The output is a 2D grid with double the number of rows.
    3. The output consists of the original grid followed by its vertical reflection.
    4. Vertical reflection means reversing the order of rows (top row becomes bottom row).

    Procedure:
    1. Take the input grid as is.
    2. Create a vertically reflected copy by reversing the row order.
    3. Concatenate the original grid with the reflected grid.
    4. Return the combined result.
    """

    # Create the reversed grid (reverse the order of rows)
    reversed_grid = grid[::-1]

    # Concatenate the original grid with the reversed grid
    result = grid + reversed_grid

    return result
