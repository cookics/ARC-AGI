def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input has cross pattern; output is same size as input
    2. Output is nested rectangular frames/corridors with specific openings
    3. Pattern creates a maze-like structure based on recursive subdivision

    Procedure:
    1. Start with all zeros
    2. Recursively draw nested rectangular corridors
    3. Each level has openings to connect to inner/outer levels
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    def draw_recursive_maze(top, left, bottom, right, level):
        """Recursively draw nested rectangular corridors."""
        if top >= bottom or left >= right or top < 0 or left < 0:
            return

        height = bottom - top + 1
        width = right - left + 1

        if height < 4 or width < 4:
            return

        # Determine opening positions based on level and size
        # Pattern alternates between different sides
        opening_row = top + height // 2
        opening_col = left + width // 2

        # Draw vertical corridors on specific columns
        col_spacing = max(2, width // 6)
        for c in [left, left + col_spacing, right - col_spacing, right]:
            if 0 <= c < cols:
                for r in range(top, bottom + 1):
                    if 0 <= r < rows:
                        # Create gaps for openings
                        if not (r == opening_row and c == left + col_spacing):
                            result[r][c] = 8

        # Draw horizontal corridors on specific rows
        row_spacing = max(2, height // 6)
        for r in [top, top + row_spacing, bottom - row_spacing, bottom]:
            if 0 <= r < rows:
                for c in range(left, right + 1):
                    if 0 <= c < cols:
                        # Create gaps for openings
                        if not (c == opening_col and r == top + row_spacing):
                            result[r][c] = 8

        # Recurse into inner region
        inner_top = top + 2
        inner_left = left + 2
        inner_bottom = bottom - 2
        inner_right = right - 2

        draw_recursive_maze(inner_top, inner_left, inner_bottom, inner_right, level + 1)

    # Start the recursive drawing
    draw_recursive_maze(0, 0, rows - 1, cols - 1, 0)

    return result
