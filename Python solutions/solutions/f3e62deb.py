def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Each input contains a 3x3 hollow square pattern (filled border, empty center).
    The pattern moves to different edges based on the value:
    - value % 3 == 0 → move to top edge (rows 0-2)
    - value % 3 == 1 → move to bottom edge (rows 7-9)
    - value % 3 == 2 → move to right edge (columns 7-9)

    Procedure:
    1. Find the 3x3 hollow square pattern and its value
    2. Determine movement direction based on value % 3
    3. Create output grid with pattern moved to appropriate edge
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find the 3x3 pattern
    pattern_value = 0
    pattern_row, pattern_col = -1, -1

    for r in range(rows - 2):
        for c in range(cols - 2):
            # Check if this is a 3x3 hollow square
            if (
                grid[r][c] != 0
                and grid[r][c]
                == grid[r][c + 1]
                == grid[r][c + 2]
                == grid[r + 1][c]
                == grid[r + 1][c + 2]
                == grid[r + 2][c]
                == grid[r + 2][c + 1]
                == grid[r + 2][c + 2]
                and grid[r + 1][c + 1] == 0
            ):
                pattern_value = grid[r][c]
                pattern_row, pattern_col = r, c
                break
        if pattern_row != -1:
            break

    assert pattern_row != -1, "Pattern not found"

    # Determine target position based on value % 3
    movement_type = pattern_value % 3

    if movement_type == 0:  # Move to top edge
        target_row, target_col = 0, pattern_col
    elif movement_type == 1:  # Move to bottom edge
        target_row, target_col = 7, pattern_col
    else:  # movement_type == 2, move to right edge
        target_row, target_col = pattern_row, 7

    # Place the pattern at the target position
    for dr in range(3):
        for dc in range(3):
            source_r, source_c = pattern_row + dr, pattern_col + dc
            target_r, target_c = target_row + dr, target_col + dc

            if 0 <= target_r < rows and 0 <= target_c < cols:
                result[target_r][target_c] = grid[source_r][source_c]

    return result
