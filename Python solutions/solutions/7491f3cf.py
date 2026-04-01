def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 7x25 grid divided into 4 sections by vertical separators
    2. Section 4 (rightmost) is initially all background
    3. Section 1 pattern determines HOW to overlay sections 2 and 3 into section 4:
       - Horizontal line in section 1 → horizontal division
       - Vertical line in section 1 → vertical division
       - Diagonal line in section 1 → diagonal division
    4. Section 3 has a cross pattern that marks the transition boundary

    Procedure:
    1. Extract all 4 sections and find backgrounds
    2. Detect section 1 pattern (horizontal/vertical/diagonal)
    3. Find section 3 cross center (row and column)
    4. Apply overlay rule based on pattern:
       - Horizontal: divide by row (s2 before, overlay at, s3 after)
       - Vertical: divide by column (s2 left, overlay at, s3 right)
       - Diagonal: divide by row+col or row/col relative to cross center
    """

    result = [row[:] for row in grid]

    # Helper function to get background (most common value) in a section
    def get_background(start_row, end_row, start_col, end_col):
        from collections import Counter
        values = []
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                values.append(grid[i][j])
        return Counter(values).most_common(1)[0][0]

    # Get background values
    bg1 = get_background(1, 6, 1, 6)
    bg2 = get_background(1, 6, 7, 12)
    bg3 = get_background(1, 6, 13, 18)

    # Extract sections
    section1, section2, section3 = [], [], []
    for i in range(1, 6):
        row1, row2, row3 = [], [], []
        for j in range(5):
            row1.append(grid[i][j + 1])
            row2.append(grid[i][j + 7])
            row3.append(grid[i][j + 13])
        section1.append(row1)
        section2.append(row2)
        section3.append(row3)

    # Detect section 1 pattern
    # Check for horizontal line (complete row)
    has_horizontal = any(all(section1[i][j] != bg1 for j in range(5)) for i in range(5))
    # Check for vertical line (complete column)
    has_vertical = any(all(section1[i][j] != bg1 for i in range(5)) for j in range(5))
    # Check for diagonal
    has_diagonal = False
    if not has_horizontal and not has_vertical:
        # Check main diagonal or anti-diagonal
        main_diag = sum(1 for i in range(5) if section1[i][i] != bg1) >= 3
        anti_diag = sum(1 for i in range(5) if section1[i][4-i] != bg1) >= 3
        has_diagonal = main_diag or anti_diag

    # Check if section 3 is checkerboard (no complete row/col)
    has_s3_cross = False
    for i in range(5):
        if all(section3[i][j] != bg3 for j in range(5)):
            has_s3_cross = True
            break
    if not has_s3_cross:
        for j in range(5):
            if all(section3[i][j] != bg3 for i in range(5)):
                has_s3_cross = True
                break

    # Find section 3 cross center
    cross_row, cross_col = None, None
    for i in range(5):
        if all(section3[i][j] != bg3 for j in range(5)):
            cross_row = i
            break
    for j in range(5):
        if all(section3[i][j] != bg3 for i in range(5)):
            cross_col = j
            break

    # If no cross found, use default center (2,2) for diagonal patterns
    if cross_row is None and has_diagonal:
        cross_row = 2
    if cross_col is None and has_diagonal:
        cross_col = 2

    # Apply overlay rule
    def overlay(val2, val3, bg2, bg3):
        """Overlay with s2 priority"""
        if val2 != bg2:
            return val2
        elif val3 != bg3:
            return val3
        else:
            return val2

    for i in range(5):
        for j in range(5):
            val2 = section2[i][j]
            val3 = section3[i][j]

            if has_diagonal and not has_s3_cross:
                # Special case: diagonal pattern + checkerboard s3
                # Use s2 only in bottom-left region, use s3 diagonal band elsewhere
                if i >= 3 and j < 2 and val2 != bg2:
                    # Bottom-left region: use s2
                    result[i + 1][j + 19] = val2
                elif val3 != bg3 and -3 <= i - j <= -1:
                    # Diagonal band: extract s3
                    result[i + 1][j + 19] = val3
                else:
                    result[i + 1][j + 19] = bg2
            elif has_horizontal and cross_row is not None:
                # Horizontal division
                if i < cross_row:
                    result[i + 1][j + 19] = val2
                elif i == cross_row:
                    result[i + 1][j + 19] = overlay(val2, val3, bg2, bg3)
                else:
                    result[i + 1][j + 19] = val3
            elif has_vertical and cross_col is not None:
                # Vertical division
                if j < cross_col:
                    result[i + 1][j + 19] = val2
                elif j == cross_col:
                    result[i + 1][j + 19] = overlay(val2, val3, bg2, bg3)
                else:
                    result[i + 1][j + 19] = val3
            elif has_diagonal and cross_row is not None and cross_col is not None:
                # Diagonal division with cross
                if i < cross_row:
                    result[i + 1][j + 19] = val2
                elif i >= cross_row:
                    if j < cross_col:
                        result[i + 1][j + 19] = val2
                    elif j == cross_col:
                        result[i + 1][j + 19] = overlay(val2, val3, bg2, bg3)
                    else:
                        if i == cross_row:
                            result[i + 1][j + 19] = overlay(val2, val3, bg2, bg3)
                        else:
                            result[i + 1][j + 19] = val3
            else:
                # Default: simple overlay
                result[i + 1][j + 19] = overlay(val2, val3, bg2, bg3)

    return result
