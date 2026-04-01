def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 10x10 grid with a header row of [1,1,2,2,3,3,4,4,5,5] and a colored border surrounding an inner area.
    2. Output keeps the header row unchanged but transforms the rest of the grid.
    3. The transformation involves finding the most frequent non-zero, non-border color within the inner area.
    4. The original border gets replaced entirely with this chosen color.
    5. A rectangular outline is drawn using the chosen color based on the bounding box of where this color originally appeared.
    6. All other inner cells are filled with 0s except for the rectangular outline.

    Procedure:
    1. Identify the border color from the second row of the grid.
    2. Extract and analyze the inner area excluding the border cells.
    3. Count frequencies of all non-zero, non-border colors found inside the border.
    4. Choose the color with the highest frequency as the replacement color.
    5. Find the bounding box coordinates of all positions where this chosen color appears.
    6. Create the output grid by replacing the border with the chosen color and drawing a rectangular outline within the bounding box area.
    """
    result = [row[:] for row in grid]  # Copy input

    # Header row stays the same
    # Border color is the color from row 1 (and all border cells)
    border_color = grid[1][0]

    # Extract inner area (excluding border)
    inner_height = len(grid) - 2
    inner_width = len(grid[0]) - 2

    # Count frequencies of colors inside border (excluding 0 and border color)
    color_count = {}
    chosen_color_positions = []

    for i in range(1, len(grid) - 1):  # Skip first and last row (border)
        for j in range(1, len(grid[0]) - 1):  # Skip first and last col (border)
            cell = grid[i][j]
            if cell != 0 and cell != border_color:
                if cell not in color_count:
                    color_count[cell] = 0
                color_count[cell] += 1

    # Choose the most frequent color
    if not color_count:
        return result  # No colors to choose from

    chosen_color = max(color_count.keys(), key=lambda x: color_count[x])

    # Find all positions of the chosen color and compute bounding box
    min_row, max_row = float("inf"), -1
    min_col, max_col = float("inf"), -1

    for i in range(1, len(grid) - 1):
        for j in range(1, len(grid[0]) - 1):
            if grid[i][j] == chosen_color:
                # Convert to inner coordinates (0-indexed within border)
                inner_row = i - 1
                inner_col = j - 1
                min_row = min(min_row, inner_row)
                max_row = max(max_row, inner_row)
                min_col = min(min_col, inner_col)
                max_col = max(max_col, inner_col)

    # Replace border with chosen color and clear inner area
    for i in range(len(result)):
        for j in range(len(result[0])):
            if i == 0:
                continue  # Keep header row
            elif i == 1 or i == len(result) - 1:
                result[i][j] = chosen_color  # Entire border rows
            elif j == 0 or j == len(result[0]) - 1:
                result[i][j] = chosen_color  # Border columns
            else:
                result[i][j] = 0  # Clear inner area

    # Create rectangular outline within bounding box
    if min_row != float("inf"):  # If we found positions
        for inner_row in range(min_row, max_row + 1):
            for inner_col in range(min_col, max_col + 1):
                # Convert back to grid coordinates
                i = inner_row + 1
                j = inner_col + 1

                # Draw rectangle outline
                if (
                    inner_row == min_row
                    or inner_row == max_row
                    or inner_col == min_col
                    or inner_col == max_col
                ):
                    result[i][j] = chosen_color

    return result
