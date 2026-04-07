def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 1s forming rectangular frame structures and 0s as background
    2. Output fills certain 0s with 2s to complete the frame structures
    3. The pattern involves:
       - Finding the bounding box of all 1s
       - Filling gaps in boundary rows between first and last 1
       - Filling gaps in boundary columns between first and last 1
       - Filling specific interior columns that have structural significance
       - Filling rows that span the full width after boundary fills

    Procedure:
    1. Find bounding box
    2. Fill gaps in boundary rows between first and last 1
    3. Fill gaps in boundary columns between first and last 1
    4. Fill special interior columns (with 1s at both boundaries and consecutive interior 1s)
    5. Fill rows that have 1s at both boundaries originally and now have non-zero at both after fills
    """

    rows = len(grid)
    cols = len(grid[0])
    result = [row[:] for row in grid]

    # Find bounding box
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                min_row = min(min_row, r)
                max_row = max(max_row, r)
                min_col = min(min_col, c)
                max_col = max(max_col, c)

    if max_row == -1:
        return result

    # Step 2: Fill gaps in boundary rows
    for boundary_row in [min_row, max_row]:
        ones = [c for c in range(cols) if grid[boundary_row][c] == 1]
        if len(ones) >= 2:
            first, last = ones[0], ones[-1]
            for c in range(first + 1, last):
                if result[boundary_row][c] == 0:
                    result[boundary_row][c] = 2

    # Step 3: Fill boundary columns - fill ALL 0s between top and bottom of bbox
    for boundary_col in [min_col, max_col]:
        for r in range(min_row, max_row + 1):
            if result[r][boundary_col] == 0:
                result[r][boundary_col] = 2

    # Step 4: Fill special interior columns
    for c in range(min_col + 1, max_col):
        # Check if column has 1s at both top and bottom boundaries
        if grid[min_row][c] == 1 and grid[max_row][c] == 1:
            # Find all 1s in this column
            ones = [r for r in range(rows) if grid[r][c] == 1]
            # Check if there are consecutive interior 1s not adjacent to boundaries
            has_interior_consecutive = False
            for i in range(len(ones) - 1):
                if ones[i+1] - ones[i] == 1:  # consecutive
                    # Check if not adjacent to boundaries
                    if ones[i] != min_row and ones[i+1] != max_row:
                        has_interior_consecutive = True
                        break

            if has_interior_consecutive and len(ones) >= 2:
                first, last = ones[0], ones[-1]
                for r in range(first + 1, last):
                    if result[r][c] == 0:
                        result[r][c] = 2

    # Step 5: Fill rows that have 1 at left boundary originally, and now have non-zero at both
    for r in range(min_row + 1, max_row):
        # Must have 1 at left boundary originally
        if grid[r][min_col] == 1:
            # And NOW must have non-zero at both boundaries after boundary fills
            if result[r][min_col] != 0 and result[r][max_col] != 0:
                # But did NOT have 1 at right boundary originally
                if grid[r][max_col] != 1:
                    # Count how many 1s this row has originally (must have enough structure)
                    orig_ones_count = sum(1 for c in range(cols) if grid[r][c] == 1)
                    if orig_ones_count >= 3:  # Row must have at least 3 1s originally
                        # Fill gaps between first and last non-zero
                        first_nonzero, last_nonzero = -1, -1
                        for c in range(cols):
                            if result[r][c] != 0:
                                if first_nonzero == -1:
                                    first_nonzero = c
                                last_nonzero = c

                        if first_nonzero != -1 and last_nonzero != -1:
                            for c in range(first_nonzero + 1, last_nonzero):
                                if result[r][c] == 0:
                                    result[r][c] = 2

    return result
