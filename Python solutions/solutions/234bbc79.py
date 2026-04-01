def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 3-row grid where value 5 acts as a separator/marker
    2. Output removes columns containing 5s and has reduced width
    3. Middle row (row 1) forms the base structure with modifications
    4. One edge row (0 or 2) becomes all zeros based on which has fewer non-zero values
    5. The other edge row gets filled with values extracted and extended from all rows
    6. Values are redistributed according to segment positions split by 5s

    Procedure:
    1. Split each row into segments using 5 as separator
    2. Extract non-zero values from each row
    3. Determine which edge row becomes zeros (fewer values) and which gets filled
    4. Calculate output width based on input width and number of 5s
    5. Build middle row by preserving base structure from row 1
    6. Build fill row by combining and extending values from all segments
    7. Build zero row as all zeros
    """

    rows, cols = len(grid), len(grid[0])

    # Count 5s per row
    fives_per_row = [sum(1 for val in grid[r] if val == 5) for r in range(rows)]

    # Split each row into segments separated by 5
    def split_by_fives(row):
        segments = []
        current = []
        for val in row:
            if val == 5:
                segments.append(current)
                current = []
            else:
                current.append(val)
        segments.append(current)
        return segments

    segments = [split_by_fives(grid[r]) for r in range(rows)]

    # Extract non-zero values
    def get_nonzero(segs):
        return [v for seg in segs for v in seg if v != 0]

    nz0 = get_nonzero(segments[0])
    nz1 = get_nonzero(segments[1])
    nz2 = get_nonzero(segments[2])

    # Determine output width based on segments structure
    num_segs_1 = sum(1 for seg in segments[1] if any(v != 0 for v in seg))

    if num_segs_1 > 2 and len(nz0) > 0:
        # Multiple segments in row 1 with values in row 0
        output_width = cols - 3
    else:
        # Standard case
        output_width = cols - max(fives_per_row)

    # Determine which edge row becomes zeros
    if len(nz0) <= len(nz2):
        zero_row, fill_row = 0, 2
    else:
        zero_row, fill_row = 2, 0

    # Build result
    result = [[0] * output_width for _ in range(rows)]

    # Build middle row: merge all segments
    pos = 0
    for seg in segments[1]:
        for val in seg:
            if pos < output_width:
                result[1][pos] = val
                pos += 1

    # Build fill row: merge values from all rows with extensions
    # Strategy: place values from middle row, then extend unique values from edge rows
    pos = 0

    # Collect unique non-zero values from each row
    unique_1 = list(dict.fromkeys(nz1))  # preserve order
    unique_edge = list(dict.fromkeys(nz0 + nz2))

    # Place middle row values first at strategic positions
    for i, val in enumerate(nz1):
        if pos < output_width:
            result[fill_row][pos] = val
            pos += 1

    # Then place and extend unique edge values
    for val in unique_edge:
        if pos < output_width and val not in unique_1:
            # Extend this value
            for _ in range(min(3, output_width - pos)):
                if pos < output_width:
                    result[fill_row][pos] = val
                    pos += 1

    return result
