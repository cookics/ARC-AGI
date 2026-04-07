def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid where each row is uniform (all cells have the same value)
    2. Consecutive rows with the same uniform value form a segment
    3. Output is a square grid with nested rectangular frames
    4. Each segment becomes a frame layer from outside to inside
    5. Position in a frame is determined by minimum distance from any edge

    Procedure:
    1. Extract segments: group consecutive uniform rows with same value
    2. Calculate output size: sum of all segment thicknesses
    3. For each position (r,c), calculate minimum distance from edges
    4. Assign color based on which segment's distance range the position falls into
    """

    # Step 1: Extract segments of consecutive uniform rows
    segments = []
    i = 0
    while i < len(grid):
        # Get the value (assuming row is uniform)
        color = grid[i][0]
        # Verify row is uniform and find consecutive rows with same value
        if all(cell == color for cell in grid[i]):
            j = i + 1
            while j < len(grid) and all(cell == color for cell in grid[j]) and grid[j][0] == color:
                j += 1
            segments.append((color, j - i))
            i = j
        else:
            i += 1

    # Step 2: Calculate output size (square grid)
    # All segments except last form frames, last fills center
    center_thickness = segments[-1][1]
    border_thickness = sum(thickness for _, thickness in segments[:-1])
    output_size = 2 * border_thickness + center_thickness

    # Step 3: Build output grid
    result = [[0] * output_size for _ in range(output_size)]

    # Step 4: Fill frames based on minimum distance from edge
    offset = 0
    for color, thickness in segments:
        for r in range(output_size):
            for c in range(output_size):
                # Calculate minimum distance from any edge
                min_dist = min(r, output_size - 1 - r, c, output_size - 1 - c)
                # Check if this position belongs to current frame
                if offset <= min_dist < offset + thickness:
                    result[r][c] = color
        offset += thickness

    return result
