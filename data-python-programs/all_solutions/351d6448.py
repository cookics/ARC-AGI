def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is divided into sections by horizontal separator rows filled with 5s
    2. Each section shows a pattern that evolves
    3. Patterns can be:
       a) Simple runs that grow in length (e.g., 2, 4, 6, 8 → 10)
       b) Runs that shift position (e.g., at cols 1, 2, 3, 4 → 5)
       c) Complex multi-row patterns like triangles that grow
    4. Output is 3 rows showing the next pattern in the sequence

    Procedure:
    1. Split input into sections separated by rows of 5s
    2. Try to find patterns in any row of each section
    3. Detect if pattern is growing, shifting, or multi-row
    4. Predict the next pattern based on detected progression
    5. Return 3-row output with predicted pattern
    """

    # Find separator rows (all 5s)
    separator_rows = []
    for i, row in enumerate(grid):
        if all(cell == 5 for cell in row):
            separator_rows.append(i)

    # Split into sections
    sections = []
    start = 0
    for sep_row in separator_rows:
        sections.append(grid[start:sep_row])
        start = sep_row + 1
    if start < len(grid):
        sections.append(grid[start:])

    if not sections or len(sections) < 2:
        return [[0] * len(grid[0]) for _ in range(3)]

    width = len(grid[0])

    # First try multi-row pattern detection (like triangular structures)
    # This should be checked before single-row patterns
    if all(len(section) == 3 for section in sections):
        # Look for a value that appears in bottom rows and grows
        for test_value in range(1, 10):
            bottom_counts = []
            valid = True

            for section in sections:
                bottom_row = section[2]
                count = 0
                for cell in bottom_row:
                    if cell == test_value:
                        count += 1
                    else:
                        break

                if count > 0:
                    bottom_counts.append(count)
                else:
                    valid = False
                    break

            if valid and len(bottom_counts) >= 2:
                diffs = [bottom_counts[i+1] - bottom_counts[i] for i in range(len(bottom_counts)-1)]
                if all(d == diffs[0] for d in diffs) and diffs[0] != 0:
                    # Found a growing triangular pattern!
                    next_count = bottom_counts[-1] + diffs[0]

                    # Build the full output by analyzing the last section structure
                    last_section = sections[-1]
                    result = [[0] * width for _ in range(3)]

                    # Copy structure from last section and extend the test_value
                    for row_idx in range(3):
                        template_row = last_section[row_idx]

                        if row_idx == 2:  # Bottom row - main growth
                            for i in range(min(next_count, width)):
                                result[row_idx][i] = test_value
                        elif row_idx == 1:  # Middle row
                            # Find where test_values start and how many consecutive ones there are
                            middle_info = []
                            for section in sections:
                                row = section[1]
                                # Find first occurrence of test_value
                                start_pos = -1
                                for j, cell in enumerate(row):
                                    if cell == test_value:
                                        start_pos = j
                                        break

                                if start_pos >= 0:
                                    # Count consecutive test_values from start_pos
                                    cnt = 0
                                    for j in range(start_pos, len(row)):
                                        if row[j] == test_value:
                                            cnt += 1
                                        else:
                                            break
                                    middle_info.append((start_pos, cnt))
                                else:
                                    middle_info.append((0, 0))

                            if middle_info and any(cnt > 0 for _, cnt in middle_info):
                                # Extract counts
                                counts = [cnt for _, cnt in middle_info]
                                # Extract starting positions (use first non-zero one)
                                start_positions = [pos for pos, cnt in middle_info if cnt > 0]
                                common_start = start_positions[0] if start_positions else 0

                                # Check if growing or capped
                                middle_diffs = [counts[i+1] - counts[i]
                                              for i in range(len(counts)-1)]
                                if len(middle_diffs) >= 2 and middle_diffs[-1] == 0:
                                    # Capped
                                    new_middle_count = counts[-1]
                                elif middle_diffs and all(d == middle_diffs[0] for d in middle_diffs):
                                    # Growing
                                    new_middle_count = counts[-1] + middle_diffs[0]
                                else:
                                    new_middle_count = counts[-1]

                                # Place the values starting from common_start
                                for i in range(min(new_middle_count, width - common_start)):
                                    result[row_idx][common_start + i] = test_value
                        else:  # Top row (row_idx == 0)
                            # Check if test_value appears in top rows
                            for col_idx, cell in enumerate(template_row):
                                if cell == test_value:
                                    result[row_idx][col_idx] = test_value

                        # Copy other non-test_value values from template
                        for col_idx in range(width):
                            if result[row_idx][col_idx] == 0 and template_row[col_idx] != 0 and template_row[col_idx] != test_value:
                                result[row_idx][col_idx] = template_row[col_idx]

                    return result

    # Try to find patterns by checking each row index across all sections
    max_rows = max(len(section) for section in sections)

    for row_idx in range(max_rows):
        # Extract this row from each section
        rows_at_idx = []
        valid = True
        for section in sections:
            if row_idx < len(section):
                rows_at_idx.append(section[row_idx])
            else:
                valid = False
                break

        if not valid:
            continue

        # First try detecting mixed pattern (full non-zero sequences that shift)
        # This should be checked before single-value patterns
        non_zero_sequences = []
        all_valid = True

        for row in rows_at_idx:
            # Find first non-zero sequence
            seq = []
            start_pos = -1
            for j, cell in enumerate(row):
                if cell != 0 and not seq:
                    start_pos = j
                    seq.append(cell)
                elif cell != 0 and seq:
                    seq.append(cell)
                elif cell == 0 and seq:
                    break

            if seq:
                non_zero_sequences.append((start_pos, tuple(seq)))
            else:
                all_valid = False
                break

        if all_valid and len(non_zero_sequences) >= 2:
            # Check if sequence is same but position shifts
            first_seq = non_zero_sequences[0][1]
            if all(s[1] == first_seq for s in non_zero_sequences):
                positions = [s[0] for s in non_zero_sequences]
                diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if all(d == diffs[0] for d in diffs) and diffs[0] != 0:
                    # Found a shifting pattern!
                    next_pos = positions[-1] + diffs[0]
                    result = [[0] * width for _ in range(3)]
                    for i, val in enumerate(first_seq):
                        if next_pos + i < width:
                            result[1][next_pos + i] = val
                    return result

        # Now try detecting growing pattern (consecutive from start)
        for test_value in range(1, 10):
            counts = []
            all_have_value = True

            for row in rows_at_idx:
                # Count consecutive occurrences from start
                count = 0
                for cell in row:
                    if cell == test_value:
                        count += 1
                    else:
                        break

                if count > 0:
                    counts.append(count)
                else:
                    all_have_value = False
                    break

            if all_have_value and len(counts) >= 2:
                # Check for arithmetic progression
                diffs = [counts[i+1] - counts[i] for i in range(len(counts)-1)]
                if all(d == diffs[0] for d in diffs) and diffs[0] != 0:
                    # Found a growing pattern!
                    next_count = counts[-1] + diffs[0]
                    result = [[0] * width for _ in range(3)]
                    for i in range(min(next_count, width)):
                        result[1][i] = test_value
                    return result

        # Try detecting position-shifting pattern for single values
        for test_value in range(1, 10):
            positions_and_patterns = []
            all_match = True

            for row in rows_at_idx:
                # Find the pattern of test_value
                pattern = []
                for cell in row:
                    if cell == test_value:
                        pattern.append(test_value)
                    elif pattern:  # We've found the end of a run
                        break

                if not pattern:
                    all_match = False
                    break

                # Find starting position
                start_pos = row.index(test_value)
                positions_and_patterns.append((start_pos, tuple(pattern)))

            if all_match and len(positions_and_patterns) >= 2:
                # Check if pattern is same but position shifts
                first_pattern = positions_and_patterns[0][1]
                if all(p[1] == first_pattern for p in positions_and_patterns):
                    positions = [p[0] for p in positions_and_patterns]
                    diffs = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    if all(d == diffs[0] for d in diffs) and diffs[0] != 0:
                        # Found a shifting pattern!
                        next_pos = positions[-1] + diffs[0]
                        result = [[0] * width for _ in range(3)]
                        for i, val in enumerate(first_pattern):
                            if next_pos + i < width:
                                result[1][next_pos + i] = val
                        return result

    # Default fallback
    return [[0] * width for _ in range(3)]
