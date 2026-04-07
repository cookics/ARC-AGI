def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing various integer values.
    2. The output is a smaller 2D rectangle extracted from inside the input grid.
    3. Each input contains exactly two parallel horizontal lines made of the same value forming a rectangular border.
    4. These horizontal lines span identical column ranges and are separated by at least one row.
    5. The content between these boundary lines (excluding the border lines themselves) becomes the output.

    Procedure:
    1. Find all unique values in the grid.
    2. For each value, find all rows where it forms a continuous horizontal line of at least 3 consecutive cells.
    3. If exactly two such rows exist with identical column ranges and are separated by at least one row, extract the content between them.
    4. Return the extracted rectangle excluding the border lines.
    """

    rows, cols = len(grid), len(grid[0])

    # Get all unique values
    unique_values = set()
    for row in grid:
        unique_values.update(row)

    # For each unique value, check if it forms two horizontal lines
    for value in unique_values:
        horizontal_lines = []

        # Find rows that are completely filled with this value (or have a continuous segment)
        for r in range(rows):
            # Find the longest continuous segment of this value in the row
            start_col, end_col = -1, -1
            for c in range(cols):
                if grid[r][c] == value:
                    if start_col == -1:
                        start_col = c
                    end_col = c
                elif start_col != -1:
                    # Break if we found a gap
                    break

            # Check if this forms a significant horizontal line (at least 3 consecutive)
            if start_col != -1 and end_col - start_col >= 2:
                # Verify the entire segment is this value
                is_complete_line = True
                for c in range(start_col, end_col + 1):
                    if grid[r][c] != value:
                        is_complete_line = False
                        break

                if is_complete_line:
                    horizontal_lines.append((r, start_col, end_col))

        # Check if we have exactly two horizontal lines with the same column range
        if len(horizontal_lines) == 2:
            r1, c1_start, c1_end = horizontal_lines[0]
            r2, c2_start, c2_end = horizontal_lines[1]

            # Lines should have the same column range and be separated by at least 1 row
            if c1_start == c2_start and c1_end == c2_end and abs(r1 - r2) > 1:
                # Extract content between the lines
                top_row = min(r1, r2)
                bottom_row = max(r1, r2)
                left_col = c1_start
                right_col = c1_end

                # Extract the rectangle content (excluding border)
                result = []
                for r in range(top_row + 1, bottom_row):
                    row_content = []
                    for c in range(left_col + 1, right_col):
                        row_content.append(grid[r][c])
                    result.append(row_content)

                return result

    # If no pattern found, return empty
    return []
