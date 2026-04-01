def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid divided vertically into left and right halves
    2. Left half has a background pattern with holes (0s)
    3. Right half contains different colored sections with non-zero values
    4. Rows completely filled with background color act as section separators
    5. Output is the left half with holes filled based on right half section colors
    6. Mapping depends on right-section sequence: decreasing→reverse, increasing→shift left, other→identity

    Procedure:
    1. Identify the background color (most common number in first column)
    2. Find section boundaries (rows completely filled with background color)
    3. Extract the left half (first half of columns)
    4. For each section, identify what number appears on the right side
    5. Create a mapping from right-side numbers to fill-numbers based on sequence pattern
    6. Fill the holes in each section with the corresponding mapped number
    """

    rows = len(grid)
    cols = len(grid[0])
    left_cols = cols // 2

    # Find background color (most common in first column)
    from collections import Counter

    background = Counter(grid[i][0] for i in range(rows)).most_common(1)[0][0]

    # Find section boundaries (rows completely filled with background)
    boundaries = []
    for i in range(rows):
        if all(grid[i][j] == background for j in range(left_cols)):
            boundaries.append(i)

    # Extract sections between boundaries
    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        if start < end:
            sections.append((start, end))

    # For each section, find the number that appears on the right side
    section_numbers = []
    for start, end in sections:
        numbers = set()
        for i in range(start, end):
            for j in range(left_cols, cols):
                if grid[i][j] != 0:
                    numbers.add(grid[i][j])
        if numbers:
            section_numbers.append(list(numbers)[0])  # Take the first non-zero number

    # Create mapping based on whether sequence is increasing or decreasing
    mapping = {}
    if len(section_numbers) >= 2:
        is_increasing = all(
            section_numbers[i] < section_numbers[i + 1]
            for i in range(len(section_numbers) - 1)
        )
        is_decreasing = all(
            section_numbers[i] > section_numbers[i + 1]
            for i in range(len(section_numbers) - 1)
        )

        if is_increasing:
            # Shift left by 1
            for i, num in enumerate(section_numbers):
                next_index = (i + 1) % len(section_numbers)
                mapping[i] = section_numbers[next_index]
        elif is_decreasing:
            # Swap first and last, keep middle unchanged
            for i, num in enumerate(section_numbers):
                if i == 0:
                    mapping[i] = section_numbers[-1]
                elif i == len(section_numbers) - 1:
                    mapping[i] = section_numbers[0]
                else:
                    mapping[i] = section_numbers[i]
        else:
            # Default case - identity mapping
            for i, num in enumerate(section_numbers):
                mapping[i] = section_numbers[i]
    else:
        # Single element case
        for i, num in enumerate(section_numbers):
            mapping[i] = section_numbers[i]

    # Create output grid (left half only)
    result = []
    for i in range(rows):
        row = grid[i][:left_cols].copy()
        result.append(row)

    # Fill holes in each section
    for section_idx, (start, end) in enumerate(sections):
        fill_number = mapping[section_idx]
        for i in range(start, end):
            for j in range(left_cols):
                if result[i][j] == 0:
                    result[i][j] = fill_number

    return result
