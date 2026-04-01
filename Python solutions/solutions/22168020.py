def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing integers where 0 represents empty space and non-zero values represent different colors.
    2. The output is a 2D grid of the same size where each non-zero color fills horizontal lines in each row.
    3. For each row, each color fills all positions between its leftmost and rightmost occurrence in that row.
    4. Colors only affect their own rows and do not interact across rows.
    5. Multiple colors can exist in the same row and each fills its own horizontal span independently.

    Procedure:
    1. Create a deep copy of the input grid to store the result.
    2. Iterate through each row of the grid.
    3. For each row, identify all non-zero colors and record their column positions.
    4. For each color in the row, find the leftmost and rightmost column positions.
    5. Fill all positions between the leftmost and rightmost positions (inclusive) with that color.
    6. Return the modified result grid.
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    for r in range(rows):
        # Find all colors and their positions in this row
        color_positions = {}
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                if color not in color_positions:
                    color_positions[color] = []
                color_positions[color].append(c)

        # For each color, fill between leftmost and rightmost positions
        for color, positions in color_positions.items():
            if positions:
                left_most = min(positions)
                right_most = max(positions)
                # Fill all positions between leftmost and rightmost
                for c in range(left_most, right_most + 1):
                    result[r][c] = color

    return result
