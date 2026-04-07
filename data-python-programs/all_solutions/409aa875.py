def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 16x16 grid with background value 7 and patterns of non-7 values
    2. Patterns in lower rows get reflected 5 rows upward with transformations
    3. For isolated columns (min distance >= 3 to other non-7 cols), place 9 five rows up
    4. For widely-spaced triplets, reflect all as 9s but middle becomes 1
    5. For repeating block patterns (8+ values in first row, 4+ clusters), mark with evenly-spaced 9s

    Procedure:
    1. Copy input grid
    2. Identify contiguous regions (consecutive rows with non-7 values)
    3. Apply region-level or row-level transformations
    """

    result = [row[:] for row in grid]
    rows = len(grid)
    cols = len(grid[0])

    def find_clusters(cols_list):
        """Find clusters of adjacent columns."""
        if not cols_list:
            return []
        clusters = []
        current = [cols_list[0]]
        for i in range(1, len(cols_list)):
            if cols_list[i] - cols_list[i-1] == 1:
                current.append(cols_list[i])
            else:
                clusters.append(current)
                current = [cols_list[i]]
        clusters.append(current)
        return clusters

    def min_distance_to_others(col_idx, cols_list):
        """Calculate minimum distance from col_idx to any other column."""
        distances = [abs(col_idx - c) for c in cols_list if c != col_idx]
        return min(distances) if distances else float('inf')

    # Identify regions (consecutive rows with non-7 values)
    regions = []
    i = 0
    while i < rows:
        if any(grid[i][j] != 7 for j in range(cols)):
            start = i
            while i < rows and any(grid[i][j] != 7 for j in range(cols)):
                i += 1
            regions.append((start, i - 1))
        else:
            i += 1

    # Track which region patterns we've seen
    seen_region_sigs = set()

    # Identify large pattern regions
    large_pattern_regions = []
    for region_start, region_end in regions:
        first_row_cols = [j for j in range(cols) if grid[region_start][j] != 7]
        first_row_clusters = find_clusters(first_row_cols)
        if len(first_row_cols) >= 8 and len(first_row_clusters) >= 4:
            large_pattern_regions.append((region_start, region_end))

    # Process each region
    for region_start, region_end in regions:
        if region_start < 5:  # Can't reflect if target would be negative
            continue

        # Get signature of this region based on first row
        first_row_cols = [j for j in range(cols) if grid[region_start][j] != 7]
        first_row_clusters = find_clusters(first_row_cols)

        # Check if this is a large repeating pattern region
        is_large_pattern = (len(first_row_cols) >= 8 and len(first_row_clusters) >= 4)

        if is_large_pattern:
            region_sig = tuple(first_row_cols)

            # Check if we've seen this pattern before
            is_first_occurrence = region_sig not in seen_region_sigs
            seen_region_sigs.add(region_sig)

            # Only create markers from the FIRST row of each large pattern occurrence
            target_row = region_start - 5
            for k in range(4):
                result[target_row][k * 3] = 9

            # Process each row in the region for cluster modification
            for row_idx in range(region_start, region_end + 1):
                if row_idx < 5:
                    continue

                non_bg_cols = [j for j in range(cols) if grid[row_idx][j] != 7]
                if not non_bg_cols:
                    continue

                clusters = find_clusters(non_bg_cols)

                # Modify second cluster only in first occurrence
                if is_first_occurrence and len(clusters) >= 2:
                    for col in clusters[1]:
                        result[row_idx][col] = 9

        else:
            # Check if this region is close to a large pattern region (within 5 rows)
            skip_region = False
            for large_start, large_end in large_pattern_regions:
                if abs(region_start - large_end) <= 5 or abs(region_end - large_start) <= 5:
                    skip_region = True
                    break

            if skip_region:
                continue

            # Process each row individually for non-large-pattern regions
            for row_idx in range(region_start, region_end + 1):
                if row_idx < 5:
                    continue

                non_bg_cols = [j for j in range(cols) if grid[row_idx][j] != 7]
                if not non_bg_cols:
                    continue

                target_row = row_idx - 5

                # Three widely-spaced values - reflect all, middle becomes 1
                if len(non_bg_cols) == 3:
                    gaps = [non_bg_cols[j+1] - non_bg_cols[j] for j in range(len(non_bg_cols)-1)]
                    if all(g > 5 for g in gaps):
                        for j in non_bg_cols:
                            result[target_row][j] = 9
                        result[target_row][non_bg_cols[1]] = 1
                    else:
                        # Reflect isolated columns (min distance >= 3)
                        for col in non_bg_cols:
                            if min_distance_to_others(col, non_bg_cols) >= 3:
                                result[target_row][col] = 9

                # Other cases: reflect isolated columns
                else:
                    for col in non_bg_cols:
                        min_dist = min_distance_to_others(col, non_bg_cols)
                        if min_dist >= 3:
                            result[target_row][col] = 9

    return result
