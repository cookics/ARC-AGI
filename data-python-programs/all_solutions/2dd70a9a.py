def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    The pattern creates L-shaped connecting paths between regions of 3s and 2s using 3s.
    Analysis shows:
    - Connect 3s region to 2s region with L-shaped paths
    - The L-path typically goes horizontal first, then vertical
    - Connection point depends on relative positions of regions


    Procedure:
    1. Find all positions containing 3s and 2s
    2. Determine connection strategy based on region positions
    3. Create L-shaped path connecting the regions
    4. Fill path with 3s, avoiding existing 8s
    """

    result = [row[:] for row in grid]  # Deep copy
    rows, cols = len(grid), len(grid[0])

    # Find all positions with 3s and 2s
    threes = []
    twos = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 3:
                threes.append((r, c))
            elif grid[r][c] == 2:
                twos.append((r, c))

    if not threes or not twos:
        return result

    # Get bounding boxes
    min_r3, max_r3 = min(r for r, c in threes), max(r for r, c in threes)
    min_c3, max_c3 = min(c for r, c in threes), max(c for r, c in threes)

    min_r2, max_r2 = min(r for r, c in twos), max(r for r, c in twos)
    min_c2, max_c2 = min(c for r, c in twos), max(c for r, c in twos)

    # Create rectangular connecting path
    # Strategy: Connect regions with rectangular frames

    if max_r3 < min_r2:
        # 3s are above 2s - choose pattern based on column overlap
        col_overlap = max(0, min(max_c3, max_c2) - max(min_c3, min_c2) + 1)

        if col_overlap > 0:
            # Columns overlap - use rectangular frame (case 3 pattern)
            extend_distance = 5
            frame_min_c = min(min_c3, min_c2) + 1  # Start one column to the right
            frame_max_c = max(max_c3, max_c2) + extend_distance

            # Top horizontal line at 3s row level
            for c in range(frame_min_c, frame_max_c + 1):
                if 0 <= min_r3 < rows and 0 <= c < cols and result[min_r3][c] == 0:
                    result[min_r3][c] = 3

            # Bottom horizontal line at 2s row level
            for c in range(frame_min_c, frame_max_c + 1):
                if 0 <= max_r2 < rows and 0 <= c < cols and result[max_r2][c] == 0:
                    result[max_r2][c] = 3

            # Right vertical line connecting top and bottom
            for r in range(min_r3, max_r2 + 1):
                if (
                    0 <= r < rows
                    and 0 <= frame_max_c < cols
                    and result[r][frame_max_c] == 0
                ):
                    result[r][frame_max_c] = 3
        else:
            # Columns don't overlap - use L-shaped connection (case 1 pattern)
            connect_row = max_r3 + 1

            # Horizontal line from 3s column range to 2s column range
            start_c = min(min_c3, min_c2)
            end_c = max(max_c3, max_c2)

            for c in range(start_c, end_c + 1):
                if (
                    0 <= connect_row < rows
                    and 0 <= c < cols
                    and result[connect_row][c] == 0
                ):
                    result[connect_row][c] = 3

            # Vertical lines to connect to 2s region
            for connect_c in [min_c2, max_c2]:
                for r in range(connect_row, min_r2):
                    if (
                        0 <= r < rows
                        and 0 <= connect_c < cols
                        and result[r][connect_c] == 0
                    ):
                        result[r][connect_c] = 3

    elif min_r3 > max_r2:
        # 3s are below 2s - connect upward with rectangular path
        # Use row closer to 2s for horizontal connection
        connect_row = max_r2 + 3  # Place horizontal line a few rows below 2s

        # Horizontal line spanning both column ranges
        start_c = min(min_c3, min_c2)
        end_c = max(max_c3, max_c2)

        for c in range(start_c, end_c + 1):
            if (
                0 <= connect_row < rows
                and 0 <= c < cols
                and result[connect_row][c] == 0
            ):
                result[connect_row][c] = 3

        # Vertical connections to both regions
        for connect_c in [min_c3, max_c3]:
            for r in range(connect_row + 1, min_r3):
                if (
                    0 <= r < rows
                    and 0 <= connect_c < cols
                    and result[r][connect_c] == 0
                ):
                    result[r][connect_c] = 3

        for connect_c in [min_c2, max_c2]:
            for r in range(max_r2 + 1, connect_row):
                if (
                    0 <= r < rows
                    and 0 <= connect_c < cols
                    and result[r][connect_c] == 0
                ):
                    result[r][connect_c] = 3
    else:
        # Overlapping rows - create rectangular frame
        # For case 3: extend the frame beyond the original regions
        extend_distance = 5  # Extend the rectangular frame

        # Determine the rectangular frame bounds
        frame_min_c = (
            min(min_c3, min_c2) - 1 if min(min_c3, min_c2) > 0 else min(min_c3, min_c2)
        )
        frame_max_c = max(max_c3, max_c2) + extend_distance
        frame_min_r = min(min_r3, min_r2)
        frame_max_r = max(max_r3, max_r2)

        # Top horizontal line
        for c in range(frame_min_c, frame_max_c + 1):
            if (
                0 <= frame_min_r < rows
                and 0 <= c < cols
                and result[frame_min_r][c] == 0
            ):
                result[frame_min_r][c] = 3

        # Bottom horizontal line
        for c in range(frame_min_c, frame_max_c + 1):
            if (
                0 <= frame_max_r < rows
                and 0 <= c < cols
                and result[frame_max_r][c] == 0
            ):
                result[frame_max_r][c] = 3

        # Right vertical line
        for r in range(frame_min_r, frame_max_r + 1):
            if (
                0 <= r < rows
                and 0 <= frame_max_c < cols
                and result[r][frame_max_c] == 0
            ):
                result[r][frame_max_c] = 3

    return result
