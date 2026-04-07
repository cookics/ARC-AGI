def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is an n×n grid with some non-zero values and zeros
    2. Output is a 2n×2n grid where each input cell becomes a 2×2 block
    3. Non-zero input cells → their 2×2 blocks are filled with that value
    4. Zero input cells → 2×2 blocks either have a diagonal of 1s or all zeros
    5. Diagonals appear on specific diagonal stripes determined by (r-c) % n
    6. Each diagonal stripe has a starting row constraint

    Procedure:
    1. Determine n and which diagonal stripes get the 1s pattern
    2. For n=2: use offset 0
    3. For odd n≥3: use offsets {0, n//2}
    4. For even n≥4: use offset {1}
    5. For each zero input cell on an active diagonal, place 1s on the 2×2 block diagonal
    """
    n = len(grid)

    # Determine which diagonal stripes to use
    if n == 2:
        diagonal_offsets = {0}
    elif n % 2 == 1:  # odd n >= 3
        diagonal_offsets = {0, n // 2}
    else:  # even n >= 4
        diagonal_offsets = {1}

    # Create output grid
    output = [[0] * (2 * n) for _ in range(2 * n)]

    # Process each input cell
    for r in range(n):
        for c in range(n):
            value = grid[r][c]
            # Output block starts at (2*r, 2*c)
            out_r, out_c = 2 * r, 2 * c

            if value != 0:
                # Fill 2×2 block with the value
                output[out_r][out_c] = value
                output[out_r][out_c + 1] = value
                output[out_r + 1][out_c] = value
                output[out_r + 1][out_c + 1] = value
            else:
                # Check if this cell should have a diagonal
                offset = (r - c) % n
                if offset in diagonal_offsets:
                    # Place diagonal of 1s in the 2×2 block
                    output[out_r][out_c] = 1
                    output[out_r + 1][out_c + 1] = 1
                # else: block remains all zeros (already initialized)

    return output
