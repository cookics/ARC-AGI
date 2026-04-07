def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 29x29 grid divided into 7x7 tiles (3x3 each) by separator lines
    2. Separator lines are at positions 0, 4, 8, 12, 16, 20, 24, 28
    3. Most tiles have one pattern (normal), some have a different pattern (special)
    4. Output adds more special tiles based on a reflection/extension pattern
    5. The pattern depends on the bounding box of special tiles:
       - For single horizontal line: extend vertically with edges
       - For multi-row: complete the pattern with specific rules

    Procedure:
    1. Extract the 7x7 tile grid by identifying which tiles are special
    2. Find the bounding box of special tiles
    3. Apply transformation rules based on bounding box shape
    4. Reconstruct the output grid with new special tiles
    """

    import copy
    result = copy.deepcopy(grid)

    # Extract tile information
    # Tiles are at rows 1-3, 5-7, 9-11, 13-15, 17-19, 21-23, 25-27
    # And columns 1-3, 5-7, 9-11, 13-15, 17-19, 21-23, 25-27
    tile_rows = [(1, 3), (5, 7), (9, 11), (13, 15), (17, 19), (21, 23), (25, 27)]
    tile_cols = [(1, 3), (5, 7), (9, 11), (13, 15), (17, 19), (21, 23), (25, 27)]

    # Extract a representative value from each tile
    # Use multiple cells to get a signature that distinguishes tile types
    def get_tile_value(r_idx, c_idx):
        r_start, r_end = tile_rows[r_idx]
        c_start, c_end = tile_cols[c_idx]
        # Return a tuple of values to better distinguish tiles
        # Use top-left, top-right, center to capture the tile pattern
        return (grid[r_start][c_start], grid[r_start][c_end], grid[r_start + 1][c_start + 1])

    # Get a tile pattern
    def get_tile_pattern(r_idx, c_idx):
        r_start, r_end = tile_rows[r_idx]
        c_start, c_end = tile_cols[c_idx]
        pattern = []
        for r in range(r_start, r_end + 1):
            pattern.append([grid[r][c] for c in range(c_start, c_end + 1)])
        return pattern

    # Set a tile pattern
    def set_tile_pattern(r_idx, c_idx, pattern):
        r_start, r_end = tile_rows[r_idx]
        c_start, c_end = tile_cols[c_idx]
        for i, r in enumerate(range(r_start, r_end + 1)):
            for j, c in enumerate(range(c_start, c_end + 1)):
                result[r][c] = pattern[i][j]

    # Find all tile types and identify special tiles
    tile_grid = [[get_tile_value(r, c) for c in range(7)] for r in range(7)]

    # Count tile types to find normal vs special
    from collections import Counter
    tile_values = [tile_grid[r][c] for r in range(7) for c in range(7)]
    value_counts = Counter(tile_values)

    # The most common value is normal, others are special
    normal_value = value_counts.most_common(1)[0][0]

    # Find special tile positions
    special_tiles = []
    for r in range(7):
        for c in range(7):
            if tile_grid[r][c] != normal_value:
                special_tiles.append((r, c))

    if not special_tiles:
        return result

    # Get patterns
    normal_pattern = None
    special_pattern = None
    for r in range(7):
        for c in range(7):
            if tile_grid[r][c] == normal_value:
                normal_pattern = get_tile_pattern(r, c)
                break
        if normal_pattern:
            break

    for r, c in special_tiles:
        special_pattern = get_tile_pattern(r, c)
        break

    # Find bounding box of special tiles
    min_r = min(r for r, c in special_tiles)
    max_r = max(r for r, c in special_tiles)
    min_c = min(c for r, c in special_tiles)
    max_c = max(c for r, c in special_tiles)

    # Determine new special tiles based on pattern
    new_special_tiles = set(special_tiles)

    height = max_r - min_r + 1
    width = max_c - min_c + 1

    # If single row of special tiles
    if height == 1:
        row = min_r
        # Extend vertically
        if row >= 3:
            # Extend upward, place original at bottom
            new_special_tiles.add((row - 1, min_c))
            new_special_tiles.add((row - 1, max_c))
            new_special_tiles.add((row - 2, min_c))
            new_special_tiles.add((row - 2, max_c))
        else:
            # Extend both ways, place original at middle
            new_special_tiles.add((row - 1, min_c))
            new_special_tiles.add((row - 1, max_c))
            new_special_tiles.add((row + 1, min_c))
            new_special_tiles.add((row + 1, max_c))
    else:
        # Multi-row pattern - extend columns to match max vertical extent
        # Group special tiles by column
        cols_with_rows = {}
        for r, c in special_tiles:
            if c not in cols_with_rows:
                cols_with_rows[c] = []
            cols_with_rows[c].append(r)

        # Find max vertical extent
        max_extent = max(max(rows) - min(rows) + 1 for rows in cols_with_rows.values())

        # Extend each column to match max extent
        for c in range(min_c, max_c + 1):
            if c in cols_with_rows:
                rows = sorted(cols_with_rows[c])
                current_extent = max(rows) - min(rows) + 1
                if current_extent < max_extent:
                    # Extend this column
                    # Alternate between extending up and down based on column position
                    if (c - min_c) % 2 == 0:
                        # Even columns extend down
                        for i in range(max_extent - current_extent):
                            new_special_tiles.add((max(rows) + i + 1, c))
                    else:
                        # Odd columns extend up
                        for i in range(max_extent - current_extent):
                            new_special_tiles.add((min(rows) - i - 1, c))

    # Debug: print special tiles
    # print(f"Original special tiles: {sorted(special_tiles)}")
    # print(f"New special tiles: {sorted(new_special_tiles)}")

    # Apply special pattern to all new special tiles
    for r, c in new_special_tiles:
        if 0 <= r < 7 and 0 <= c < 7:
            set_tile_pattern(r, c, special_pattern)

    return result
