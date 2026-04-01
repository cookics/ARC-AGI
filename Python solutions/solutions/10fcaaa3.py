def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input H×W becomes output 2H×2W by tiling in 2x2 pattern
    2. Non-zero values are preserved
    3. For zeros in rows with non-zero + all-zero columns: (i+j+is_bottom)%2==1 → 8, else 0
    4. For zeros in rows with non-zero + non-all-zero columns: stay 0
    5. For all-zero rows sandwiched: all 8s
    6. For all-zero rows adjacent: if adjacent row has non-zero at column → 0, else check if all-zero column → left=8/right=0, else 8
    7. For all-zero rows not adjacent: checkerboard (i+j)%2==0 → 8

    Procedure:
    1. Tile input in 2x2, apply context-dependent fill rules
    """
    H, W = len(grid), len(grid[0])
    result = [[0] * (2 * W) for _ in range(2 * H)]

    # Precompute properties
    col_all_zero = [all(grid[i][j] == 0 for i in range(H)) for j in range(W)]
    row_has_nonzero = [any(grid[i][j] != 0 for j in range(W)) for i in range(H)]

    for out_i in range(2 * H):
        for out_j in range(2 * W):
            in_i = out_i % H
            in_j = out_j % W
            val = grid[in_i][in_j]

            if val != 0:
                result[out_i][out_j] = val
            elif row_has_nonzero[in_i]:
                # Row has non-zero
                if col_all_zero[in_j]:
                    # All-zero column: apply top/bottom rule
                    is_bottom = (out_i >= H)
                    if (in_i + in_j + (1 if is_bottom else 0)) % 2 == 1:
                        result[out_i][out_j] = 8
                    else:
                        result[out_i][out_j] = 0
                else:
                    result[out_i][out_j] = 0
            else:
                # Row is all-zero
                has_nonzero_above = in_i > 0 and row_has_nonzero[in_i - 1]
                has_nonzero_below = in_i < H - 1 and row_has_nonzero[in_i + 1]

                if has_nonzero_above and has_nonzero_below:
                    result[out_i][out_j] = 8
                elif has_nonzero_above or has_nonzero_below:
                    adj_row = in_i - 1 if has_nonzero_above else in_i + 1

                    if grid[adj_row][in_j] != 0:
                        result[out_i][out_j] = 0
                    elif col_all_zero[in_j]:
                        # All-zero column: apply left/right rule
                        is_right = (out_j >= W)
                        result[out_i][out_j] = 8 if not is_right else 0
                    else:
                        result[out_i][out_j] = 8
                else:
                    if (in_i + in_j) % 2 == 0:
                        result[out_i][out_j] = 8
                    else:
                        result[out_i][out_j] = 0

    return result
