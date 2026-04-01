def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 6×6 grid with values 0 and 8
    2. Output is a 6×6 grid with values 0, 2, and 5
    3. 0s remain as 0s
    4. 8s are colored based on position within 3×2 blocks
    5. Pattern uses (r%3, c%2) and block coordinates

    Procedure:
    1. Divide grid into 3×2 blocks (3 rows × 2 cols each)
    2. Use block position and local position to determine color
    3. Apply flip based on top-left corner value
    """
    rows = len(grid)
    cols = len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Determine flip based on top-left value
    flip = (grid[0][0] == 0)

    # Process each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                result[r][c] = 0
            else:  # grid[r][c] == 8
                # 3×2 block structure
                br = r // 3  # block row (0 or 1)
                bc = c // 2  # block column (0, 1, or 2)
                lr = r % 3   # local row within block (0, 1, or 2)
                lc = c % 2   # local col within block (0 or 1)

                # Determine base color using a weighted formula
                base = (lr + lc + br * 2 + bc * 2) % 2

                # Apply flip
                if flip:
                    color = 5 if base == 0 else 2
                else:
                    color = 2 if base == 0 else 5

                result[r][c] = color

    return result
