def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid where some rows contain repeating patterns
    2. Some elements in these patterns are corrupted/anomalous
    3. Output is the same grid with patterns corrected
    4. Border elements (first 2 and last 2 columns) are preserved
    5. Each row may have a different repeating pattern period (e.g., [1,3] or [3,3,4])

    Procedure:
    1. For each row, extract the middle part (columns 2 to -2)
    2. Try different period lengths (1, 2, 3, ...) to find repeating pattern
    3. For each period, use majority voting at each pattern position
    4. Select the period with highest match rate
    5. Reconstruct the row using the detected pattern
    """
    from collections import Counter

    result = [row[:] for row in grid]

    for i in range(len(grid)):
        row = grid[i]
        if len(row) < 5:  # Too short to have a meaningful pattern
            continue

        # Extract middle part (skip border columns)
        middle = row[2:-2]
        if len(middle) == 0:
            continue

        # Skip rows with no variation (all same value)
        if len(set(middle)) <= 1:
            continue

        # Try different period lengths and find the best one
        best_period = None
        best_match_rate = 0

        for period in range(1, min(len(middle) // 2 + 1, 20)):
            # For this period, count how many elements match the most common value at each position
            total_matches = 0

            for pos in range(period):
                # Collect all values at this pattern position
                values = [middle[j] for j in range(pos, len(middle), period)]
                # Count matches with most common value
                most_common_count = Counter(values).most_common(1)[0][1]
                total_matches += most_common_count

            match_rate = total_matches / len(middle)

            if match_rate > best_match_rate:
                best_match_rate = match_rate
                best_period = period

        if best_period is None:
            continue

        # Reconstruct the pattern using the best period
        pattern = []
        for pos in range(best_period):
            values = [middle[j] for j in range(pos, len(middle), best_period)]
            most_common = Counter(values).most_common(1)[0][0]
            pattern.append(most_common)

        # Build new middle using the repeating pattern
        new_middle = [pattern[j % best_period] for j in range(len(middle))]

        # Update result if the pattern was corrected
        if new_middle != middle:
            result[i] = row[:2] + new_middle + row[-2:]

    return result
