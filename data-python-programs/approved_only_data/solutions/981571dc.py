def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 30x30 grid with some cells containing 0 (representing missing/corrupted data)
    2. Output is the same grid with 0s filled in
    3. The output grid has both vertical and horizontal reflection symmetry
    4. Vertical symmetry: position (i, j) mirrors to position (i, 31-j) within each row
    5. Horizontal symmetry: position (i, j) mirrors to position (31-i, j) across rows
    6. Note: The grid is 30x30 (indices 0-29), but the mirror formula uses 31, so:
       - Rows 0-1 and columns 0-1 don't have mirrors within bounds (31-0=31, 31-1=30)
       - But all 0s appear in positions where at least one mirror exists within bounds

    Procedure:
    1. Copy all non-zero cells from input to output
    2. Iteratively fill 0s using symmetry:
       a. Try vertical mirror: copy from position (i, 31-j) if non-zero
       b. Try horizontal mirror: copy from position (31-i, j) if non-zero
       c. Try diagonal mirror: copy from position (31-i, 31-j) if non-zero
    3. Repeat until all 0s are filled
    """

    n = len(grid)
    result = [row[:] for row in grid]  # Deep copy

    # Iteratively fill 0s using symmetry
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(n):
                if result[i][j] == 0:
                    # Try vertical mirror within the same row
                    mirror_j = 31 - j
                    if 0 <= mirror_j < n and result[i][mirror_j] != 0:
                        result[i][j] = result[i][mirror_j]
                        changed = True
                        continue

                    # Try horizontal mirror across rows
                    mirror_i = 31 - i
                    if 0 <= mirror_i < n and result[mirror_i][j] != 0:
                        result[i][j] = result[mirror_i][j]
                        changed = True
                        continue

                    # Try diagonal mirror (both horizontal and vertical)
                    if 0 <= mirror_i < n and 0 <= mirror_j < n and result[mirror_i][mirror_j] != 0:
                        result[i][j] = result[mirror_i][mirror_j]
                        changed = True

    return result
