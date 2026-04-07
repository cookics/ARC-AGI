def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with exactly three distinct color values
    2. Each row has a consistent left boundary color (first column) and right boundary color (last column)
    3. The interior contains mostly a background color (7) with scattered instances of the boundary colors
    4. Output shows boundary colors gravitating toward their respective edges through bubble-sort-like movement
    5. Left boundary color moves leftward, right boundary color moves rightward, swapping only with background
    6. Movement continues until convergence (no more swaps possible)

    Procedure:
    1. Identify all three colors from the grid
    2. Determine left boundary color (first column value), right boundary color (last column value)
    3. Determine background color (the remaining third color)
    4. For each row independently:
       - Iteratively swap left boundary color leftward with adjacent background cells
       - Iteratively swap right boundary color rightward with adjacent background cells
       - Repeat until no more swaps are possible (convergence)
    5. Return the transformed grid
    """

    # Find all unique values in the grid
    all_colors = set()
    for row in grid:
        all_colors.update(row)

    # Identify boundary colors from the first row
    left_boundary_color = grid[0][0]
    right_boundary_color = grid[0][-1]

    # Background color is the remaining color
    background_color = (all_colors - {left_boundary_color, right_boundary_color}).pop()

    result = []

    for row in grid:
        new_row = list(row)

        # Iterate until no more swaps are possible (convergence)
        changed = True
        while changed:
            changed = False

            # Move left boundary color leftward
            for i in range(len(new_row) - 1):
                if (
                    new_row[i] == background_color
                    and new_row[i + 1] == left_boundary_color
                ):
                    new_row[i], new_row[i + 1] = new_row[i + 1], new_row[i]
                    changed = True

            # Move right boundary color rightward
            for i in range(len(new_row) - 1, 0, -1):
                if (
                    new_row[i] == background_color
                    and new_row[i - 1] == right_boundary_color
                ):
                    new_row[i], new_row[i - 1] = new_row[i - 1], new_row[i]
                    changed = True

        result.append(new_row)

    return result
