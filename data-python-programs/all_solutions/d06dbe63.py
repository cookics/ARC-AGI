def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 13×13 grid of zeros with a single '8' at some position
    2. Output keeps the '8' and adds diagonal staircase patterns using '5's
    3. Upward staircase: alternates single vertical steps up with horizontal triplets (3 cells) going right
    4. Downward staircase: alternates single vertical steps down with horizontal triplets, shifting left by 2 each time

    Procedure:
    1. Find position of '8'
    2. Create upward stair pattern
    3. Create downward stair pattern
    4. Return modified grid
    """

    result = [row[:] for row in grid]
    rows, cols = len(grid), len(grid[0])

    # Find the position of '8'
    eight_r, eight_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 8:
                eight_r, eight_c = r, c
                break
        if eight_r is not None:
            break

    assert eight_r is not None, "Could not find '8' in the grid"

    # Create upward stairs
    r, c = eight_r, eight_c
    step = 1
    while True:
        if step % 2 == 1:  # Odd steps: single vertical step up
            r -= 1
            if r < 0:
                break
            result[r][c] = 5
        else:  # Even steps: horizontal line of 3 to the right
            r -= 1
            if r < 0:
                break
            for i in range(3):
                if c + i < cols:
                    result[r][c + i] = 5
            c += 2  # Move to end of horizontal line
        step += 1

    # Create downward stairs
    r, c = eight_r, eight_c
    step = 1
    while True:
        if step % 2 == 1:  # Odd steps: single vertical step down
            r += 1
            if r >= rows:
                break
            result[r][c] = 5
        else:  # Even steps: horizontal line of 3, shifted left by 2
            c_new = c - 2  # Shift left by 2 for the horizontal line
            # Check if we can place at least the leftmost cell
            if c_new >= 0:
                r += 1
                if r >= rows:
                    break
                for i in range(3):
                    if c_new + i >= 0 and c_new + i < cols:
                        result[r][c_new + i] = 5
                c = c_new  # Update column position
            else:
                # Can't place horizontal line, but try one more single step
                r += 1
                if r >= rows:
                    break
                result[r][c] = 5
                break  # Stop after this final single step
        step += 1

    return result
