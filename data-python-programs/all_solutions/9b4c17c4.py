def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input grid contains regions with background colors 1 and 8.
    2. Value 2 appears scattered within these background regions.
    3. The transformation preserves the count of 2s within each region.
    4. In regions with background color 8, all 2s are moved to the leftmost positions.
    5. In regions with background color 1, all 2s are moved to the rightmost positions.
    6. The transformation is applied row by row independently.

    Procedure:
    1. Process each row of the grid individually.
    2. Identify contiguous regions by their background color, ignoring 2s.
    3. For each region, count the number of 2s present.
    4. Clear all 2s from the region by replacing them with the background color.
    5. Place the counted 2s at the appropriate edge based on background color.
    6. For background 8, place 2s starting from the leftmost position.
    7. For background 1, place 2s starting from the rightmost position.
    """

    result = []

    for row in grid:
        new_row = row[:]

        # Find regions by looking at non-2 values to determine background
        regions = []
        i = 0
        while i < len(row):
            if row[i] != 2:
                # Start of a new region
                start = i
                bg = row[i]
                # Find end of this region (continues while same background or 2s)
                while i < len(row) and (row[i] == bg or row[i] == 2):
                    i += 1
                regions.append((start, i - 1, bg))
            else:
                i += 1

        # For each region, collect 2s and move them to appropriate edge
        for start, end, bg in regions:
            # Count 2s in this region
            twos_count = sum(1 for i in range(start, end + 1) if row[i] == 2)

            # Clear 2s from this region (replace with background)
            for i in range(start, end + 1):
                if new_row[i] == 2:
                    new_row[i] = bg

            # Place 2s at appropriate edge
            if bg == 1:  # move to right edge
                for i in range(twos_count):
                    new_row[end - i] = 2
            elif bg == 8:  # move to left edge
                for i in range(twos_count):
                    new_row[start + i] = 2

        result.append(new_row)

    return result
