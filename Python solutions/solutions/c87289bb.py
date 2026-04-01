def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with vertical lines of 8s in the top portion
    2. One row contains 2s forming one or more horizontal segments
    3. Output draws rectangular boxes around the 2s segments
    4. Boxes are formed by extending vertical walls downward and adding horizontal segments
    5. Existing vertical 8 columns outside red segments are preserved
    6. New boundary columns are added adjacent to red segments

    Procedure:
    1. Find the row containing 2s
    2. Identify existing vertical 8 columns from rows above the 2s
    3. Find connected segments of 2s in that row
    4. Determine new boundary columns for each segment
    5. Fill the row above 2s with existing columns plus new boundaries
    6. Fill from 2s row downward with the new column structure
    """
    result = [row[:] for row in grid]

    # Find row with red cells
    red_row_idx = -1
    for i, row in enumerate(grid):
        if 2 in row:
            red_row_idx = i
            break

    if red_row_idx == -1:
        return result

    # Find existing columns
    existing_columns = set()
    for col in range(len(grid[0])):
        for row in range(red_row_idx):
            if grid[row][col] == 8:
                existing_columns.add(col)
                break

    # Find red positions
    red_positions = [
        col for col in range(len(grid[red_row_idx])) if grid[red_row_idx][col] == 2
    ]

    if not red_positions:
        return result

    # Determine new column structure based on examples
    leftmost_red = min(red_positions)
    rightmost_red = max(red_positions)

    new_columns = set()

    # Keep existing columns outside red bounding box
    for col in existing_columns:
        if col < leftmost_red or col > rightmost_red:
            new_columns.add(col)

    # Add left boundary if there's an existing column at leftmost position
    if leftmost_red > 0 and leftmost_red in existing_columns:
        new_columns.add(leftmost_red - 1)

    # Always add right boundary
    if rightmost_red < len(grid[0]) - 1:
        new_columns.add(rightmost_red + 1)

    # For multi-segment cases, add left boundary of subsequent segments
    segments = []
    start = red_positions[0]
    for i in range(1, len(red_positions)):
        if red_positions[i] != red_positions[i - 1] + 1:
            segments.append((start, red_positions[i - 1]))
            start = red_positions[i]
    segments.append((start, red_positions[-1]))

    if len(segments) > 1:
        for i in range(1, len(segments)):
            seg_start, _ = segments[i]
            prev_seg_end = segments[i - 1][1]
            gap_size = seg_start - prev_seg_end - 1

            # Only add boundary if gap is larger than 1
            if gap_size > 1 and seg_start > 0:
                new_columns.add(seg_start - 1)

    # Fill row above red with existing columns + new boundary columns
    if red_row_idx > 0:
        # Keep existing columns in row above red
        for col in existing_columns:
            if result[red_row_idx - 1][col] == 0:
                result[red_row_idx - 1][col] = 8

            # For existing columns within red segments, extend one position right
            # Only if the column is not at the start of a red segment
            if leftmost_red <= col <= rightmost_red:
                # Check if this column is at the start of any red segment
                is_segment_start = False
                for start, end in segments:
                    if col == start:
                        is_segment_start = True
                        break

                if (
                    not is_segment_start
                    and col + 1 <= rightmost_red
                    and result[red_row_idx - 1][col + 1] == 0
                ):
                    result[red_row_idx - 1][col + 1] = 8

        # Add new boundary columns in row above red
        for col in new_columns:
            if col not in existing_columns and result[red_row_idx - 1][col] == 0:
                result[red_row_idx - 1][col] = 8

    # Fill red row and below with only new column structure
    for row_idx in range(red_row_idx, len(grid)):
        for col in new_columns:
            if result[row_idx][col] != 2:  # Don't overwrite red cells
                result[row_idx][col] = 8

    return result
