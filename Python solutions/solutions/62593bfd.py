def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with a dominant background value and several colored objects
    2. Output is the same grid with objects moved to either top or bottom
    3. Objects preserve their column positions and shapes during movement
    4. Movement direction depends on object color value relative to background
    5. For background=0: odd values move to top, even values move to bottom
    6. For background>0: values>background move to top, values<background follow modulo-3 pattern

    Procedure:
    1. Find the background value (most common value in the grid)
    2. Identify all non-background objects and their cell positions
    3. Classify each object value as "top" or "bottom" based on background rules
    4. Place top objects starting from row 0, preserving their relative row offsets
    5. Place bottom objects ending at last row, preserving their relative row offsets
    """

    rows, cols = len(grid), len(grid[0])

    # Find background value (most common)
    value_counts = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            value_counts[val] = value_counts.get(val, 0) + 1

    background = max(value_counts.keys(), key=lambda x: value_counts[x])

    # Find all objects (non-background values) and their positions
    objects = {}
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val != background:
                if val not in objects:
                    objects[val] = []
                objects[val].append((r, c))

    # Create result grid filled with background
    result = [[background for _ in range(cols)] for _ in range(rows)]

    # Determine which objects go to top vs bottom
    top_objects = []
    bottom_objects = []

    for value in objects:
        if background == 0:
            # Rule for background 0: odd to top, even to bottom
            if value % 2 == 1:
                top_objects.append(value)
            else:
                bottom_objects.append(value)
        else:
            # Rule for non-zero background
            if value > background:
                top_objects.append(value)
            else:
                # For values < background, use pattern from example 2
                if value % 3 == 1:  # 1,4 pattern
                    bottom_objects.append(value)
                else:  # 2,3 pattern
                    top_objects.append(value)

    # Place top objects - they all start from row 0
    for value in top_objects:
        positions = objects[value]

        # Find the original bounding box
        min_row = min(r for r, c in positions)

        # Place at top, starting from row 0
        for r, c in positions:
            new_row = r - min_row  # Shift to start from row 0
            result[new_row][c] = value

    # Place bottom objects - they all end at the bottom
    for value in bottom_objects:
        positions = objects[value]

        # Find the original bounding box
        max_row = max(r for r, c in positions)

        # Place at bottom, ending at last row
        for r, c in positions:
            new_row = rows - 1 - (max_row - r)  # Shift to end at bottom
            result[new_row][c] = value

    return result
