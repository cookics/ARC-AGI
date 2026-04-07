def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 15x15 grid containing mostly zeros with scattered non-zero marker values.
    2. Each non-zero marker creates a horizontal zone in the output grid filled with that marker's value.
    3. Zones are divided by calculating midpoints between consecutive marker row positions.
    4. Within each zone, certain rows get completely filled while others get only edge fills.
    5. Marker rows, top boundary (row 0), and bottom boundary (row 14) get full horizontal lines.
    6. All other rows in each zone get filled only at the leftmost and rightmost columns.

    Procedure:
    1. Scan the input grid to find all non-zero marker values and their row positions.
    2. Sort the markers by their row positions to process them in vertical order.
    3. Calculate zone boundaries for each marker using midpoint formula between adjacent markers.
    4. For each zone, fill rows with the marker value using appropriate pattern.
    5. Apply full row fills for marker rows and grid boundaries within the zone.
    6. Apply edge-only fills for all other rows within the zone.
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all non-zero markers
    markers = []
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                markers.append((r, grid[r][c]))

    assert len(markers) > 0, "Should have at least one marker"

    # Sort markers by row position
    markers.sort(key=lambda x: x[0])

    # Calculate zone boundaries
    zones = []
    marker_rows = [m[0] for m in markers]
    marker_values = [m[1] for m in markers]

    for i in range(len(markers)):
        if len(markers) == 1:
            # Single marker covers entire grid
            start_row = 0
            end_row = rows - 1
        elif i == 0:
            # First zone: from 0 to midpoint with next marker
            start_row = 0
            end_row = (marker_rows[0] + marker_rows[1]) // 2
        elif i == len(markers) - 1:
            # Last zone: from midpoint with previous marker + 1 to end
            start_row = (marker_rows[i - 1] + marker_rows[i]) // 2 + 1
            end_row = rows - 1
        else:
            # Middle zone: from midpoint with previous + 1 to midpoint with next
            start_row = (marker_rows[i - 1] + marker_rows[i]) // 2 + 1
            end_row = (marker_rows[i] + marker_rows[i + 1]) // 2

        zones.append((start_row, end_row, marker_values[i], marker_rows[i]))

    # Fill each zone
    for start_row, end_row, value, marker_row in zones:
        for r in range(start_row, end_row + 1):
            if (
                r == marker_row  # Marker row gets full line
                or r == 0  # Top boundary gets full line
                or r == rows - 1
            ):  # Bottom boundary gets full line
                # Fill entire row
                for c in range(cols):
                    result[r][c] = value
            else:
                # Fill only edges (left and right columns)
                result[r][0] = value
                result[r][cols - 1] = value

    return result
