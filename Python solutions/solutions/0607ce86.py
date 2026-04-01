def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid with repeating tile patterns that contain noise (scattered wrong values)
    2. Output has noise cleaned via majority voting across pattern repetitions
    3. The grid has periodic structure with patterns repeating horizontally and vertically
    4. Rows and columns that are mostly zeros in input become entirely zeros in output (padding regions)
    5. The main pattern area has consistent tile repetitions, outside is all zeros

    Procedure:
    1. Identify padding rows and columns (those mostly filled with zeros in input)
    2. Try different tile periods to find the best repeating pattern
    3. Use majority voting across tile repetitions to clean noise
    4. Force padding rows and columns to be all zeros in output
    5. Return the cleaned grid
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        return grid

    from collections import Counter

    # Identify rows and columns that should be all zeros (padding regions)
    # Use multiple criteria to detect padding
    zero_threshold_high = 0.85  # Very high zero density
    zero_threshold_low = 0.60  # Moderate zero density

    padding_rows = set()
    for r in range(rows):
        zero_count = sum(1 for c in range(cols) if grid[r][c] == 0)
        zero_ratio = zero_count / cols
        # Mark as padding if very high zero ratio
        # OR if moderate zero ratio and at the edge of the grid (beginning or end)
        at_edge = (r < 1) or (r >= rows - 3)
        if zero_ratio >= zero_threshold_high or (
            zero_ratio >= zero_threshold_low and at_edge
        ):
            padding_rows.add(r)

    padding_cols = set()
    for c in range(cols):
        zero_count = sum(1 for r in range(rows) if grid[r][c] == 0)
        zero_ratio = zero_count / rows
        # Mark as padding if very high zero ratio
        # OR if moderate zero ratio and at the edge of the grid (beginning or end)
        # Include column 0 and 1 as potential edge columns
        at_edge = (c <= 1) or (c >= cols - 4)
        if zero_ratio >= zero_threshold_high or (
            zero_ratio >= zero_threshold_low and at_edge
        ):
            padding_cols.add(c)

    # Try different periods to find the best fit
    best_result = None
    best_score = 0

    # Test various period sizes
    for period_h in range(5, min(rows + 1, 12)):
        for period_w in range(5, min(cols + 1, 12)):
            # Create result grid using majority voting
            result = [[0] * cols for _ in range(rows)]

            # For each position in the grid
            for r in range(rows):
                for c in range(cols):
                    # Skip if this is in a padding region
                    if r in padding_rows or c in padding_cols:
                        result[r][c] = 0
                        continue

                    # Find the offset within the period
                    r_offset = r % period_h
                    c_offset = c % period_w

                    # Collect all values at this offset position across all repetitions
                    values = []
                    for rr in range(r_offset, rows, period_h):
                        for cc in range(c_offset, cols, period_w):
                            # Skip values from padding regions
                            if rr not in padding_rows and cc not in padding_cols:
                                values.append(grid[rr][cc])

                    # Use majority vote to determine the canonical value
                    # In case of ties, prefer values that appear more frequently overall in non-padding regions
                    if values:
                        counter = Counter(values)
                        # Get top candidates (those tied for most common)
                        max_count = counter.most_common(1)[0][1]
                        candidates = [
                            val for val, cnt in counter.items() if cnt == max_count
                        ]

                        if len(candidates) == 1:
                            result[r][c] = candidates[0]
                        else:
                            # Tie-breaker: prefer the value that is more common in the overall grid
                            # Count each candidate's frequency in the entire non-padding region
                            global_counts = {}
                            for candidate in candidates:
                                count = 0
                                for rr in range(rows):
                                    for cc in range(cols):
                                        if (
                                            rr not in padding_rows
                                            and cc not in padding_cols
                                        ):
                                            if grid[rr][cc] == candidate:
                                                count += 1
                                global_counts[candidate] = count
                            result[r][c] = max(
                                candidates, key=lambda v: global_counts[v]
                            )

            # Calculate consistency score for this period
            # Higher score means more cells agree with the most common value
            score = 0
            total = 0
            for r in range(rows):
                for c in range(cols):
                    if r in padding_rows or c in padding_cols:
                        continue

                    r_offset = r % period_h
                    c_offset = c % period_w

                    values = []
                    for rr in range(r_offset, rows, period_h):
                        for cc in range(c_offset, cols, period_w):
                            if rr not in padding_rows and cc not in padding_cols:
                                values.append(grid[rr][cc])

                    if values:
                        counter = Counter(values)
                        most_common_count = counter.most_common(1)[0][1]
                        score += most_common_count
                        total += len(values)

            consistency = score / total if total > 0 else 0

            # Keep the result with the highest consistency
            if consistency > best_score:
                best_score = consistency
                best_result = result

    return best_result if best_result is not None else grid
