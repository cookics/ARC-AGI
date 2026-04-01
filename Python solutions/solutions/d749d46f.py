def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is h×w grid, output is always 10 rows
    2. Creates L-shaped combination: rotated input (vertical part) + original input (horizontal part)
    3. Rotation: flip vertically then rotate 90° clockwise
    4. Left h columns contain rotated input (trimmed to 10 rows)
    5. Top h rows, from column h onward, contain input (skipping first column)
    6. Overlap regions use sparse color if present, otherwise dominant color
    7. Remaining areas filled with dominant color

    Procedure:
    1. Find dominant (most frequent) color
    2. Flip input vertically, then rotate 90° clockwise to get vertical component
    3. Calculate output dimensions
    4. Place rotated input in left h columns (first 10 rows)
    5. Place original input (columns 1 to w-1) starting from column h
    6. Handle overlap with sparse color priority
    7. Fill remaining with dominant color
    """
    from collections import Counter

    h, w = len(grid), len(grid[0])

    # Find dominant color (most frequent)
    all_values = [val for row in grid for val in row]
    color_counts = Counter(all_values)
    dominant = color_counts.most_common(1)[0][0]
    # Sparse color is the other one
    sparse = [c for c in color_counts if c != dominant][0] if len(color_counts) > 1 else dominant

    # Rotate 90° clockwise: column i becomes row i (read bottom to top)
    # For CW: new_grid[r][c] = old_grid[h-1-c][r]
    rotated = []
    for r in range(w):
        row = []
        for c in range(h):
            row.append(grid[h - 1 - c][r])
        rotated.append(row)
    # rotated is now w rows × h columns

    # Calculate output dimensions
    out_h = 10

    # Output width formula: h + (w-1) + extra rotated rows
    # Extra rotated rows = max(0, w - 11)
    out_w = h + w - 1 + max(0, w - 11)

    # Initialize with dominant color
    result = [[dominant] * out_w for _ in range(out_h)]

    # Place first 10 rows of rotated in left h columns
    for r in range(min(len(rotated), out_h)):
        for c in range(h):
            result[r][c] = rotated[r][c]

    # Place original input in top h rows, starting from column h
    for input_row in range(h):
        for input_col in range(w):
            # Place in output columns h onwards
            col_idx = h + input_col
            if col_idx < out_w:
                result[input_row][col_idx] = grid[input_row][input_col]

    # Place extra rotated rows (beyond row 10) as horizontal extensions
    # These go in columns starting after h + w - 1
    if w > 10:
        extra_start_col = h + w - 1
        for extra_row_idx in range(10, min(w, 10 + (w - 10))):
            rotated_row = rotated[extra_row_idx]
            col_offset = extra_start_col + (extra_row_idx - 10) * h
            for c in range(h):
                out_col = col_offset + c
                if out_col < out_w:
                    # Place in row (extra_row_idx - 10)
                    out_row = extra_row_idx - 10
                    if out_row < out_h:
                        result[out_row][out_col] = rotated_row[c]

    return result
