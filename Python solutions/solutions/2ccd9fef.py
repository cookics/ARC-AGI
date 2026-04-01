def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains repeating sections (either row-wise or column-wise)
    2. Within each section, colored regions grow in width across sections
    3. Output extrapolates to the next iteration by continuing growth pattern
    4. One edge stays fixed while the other extends

    Procedure:
    1. Try vertical sections first, then horizontal (transposed)
    2. Find section boundaries by checking for matching frame patterns
    3. For each row, track contiguous spans of each color across sections
    4. Determine which edge is fixed and extrapolate the growth
    """
    from collections import Counter

    def find_color_spans(row, bg):
        """Find all contiguous spans of non-background colors"""
        spans = []
        i = 0
        while i < len(row):
            if row[i] != bg:
                color = row[i]
                start = i
                while i < len(row) and row[i] == color:
                    i += 1
                spans.append((color, start, i - 1))
            else:
                i += 1
        return spans

    def extrapolate_span(starts, ends):
        """Determine the next span based on growth pattern"""
        if len(starts) == 1:
            return starts[0], ends[0]

        # Check which edge is fixed
        if len(set(starts)) == 1:
            # Left edge fixed, right extends
            next_start = starts[0]
            if len(ends) >= 2:
                growth = ends[-1] - ends[-2]
                next_end = ends[-1] + growth
            else:
                next_end = ends[-1]
        elif len(set(ends)) == 1:
            # Right edge fixed, left extends
            next_end = ends[0]
            if len(starts) >= 2:
                growth = starts[-1] - starts[-2]
                next_start = starts[-1] + growth
            else:
                next_start = starts[-1]
        else:
            # Both edges moving
            if len(starts) >= 2:
                next_start = starts[-1] + (starts[-1] - starts[-2])
            else:
                next_start = starts[-1]

            if len(ends) >= 2:
                next_end = ends[-1] + (ends[-1] - ends[-2])
            else:
                next_end = ends[-1]

        return next_start, next_end

    def extrapolate_row(section_spans, result_row, bg):
        """Match spans across sections and extrapolate to result row"""
        if not section_spans:
            return

        num_sections = len(section_spans)
        used = [set() for _ in range(num_sections)]

        # For each span in each section, try to match across sections
        for sec_idx in range(num_sections):
            for span_idx, (color, start, end) in enumerate(section_spans[sec_idx]):
                if span_idx in used[sec_idx]:
                    continue

                # Find matching spans in other sections
                matched = [(sec_idx, start, end)]
                used[sec_idx].add(span_idx)

                for other_sec in range(num_sections):
                    if other_sec == sec_idx:
                        continue
                    for other_idx, (other_color, other_start, other_end) in enumerate(section_spans[other_sec]):
                        if other_idx in used[other_sec] or other_color != color:
                            continue

                        # Check if spans match (similar position or overlap)
                        if (abs(other_start - start) <= 5 or abs(other_end - end) <= 5 or
                            (other_start <= end and other_end >= start)):
                            matched.append((other_sec, other_start, other_end))
                            used[other_sec].add(other_idx)
                            break

                # Extrapolate based on matched spans
                matched.sort(key=lambda x: x[0])
                starts = [s[1] for s in matched]
                ends = [s[2] for s in matched]

                next_start, next_end = extrapolate_span(starts, ends)

                # Fill result
                for c in range(max(0, next_start), min(len(result_row), next_end + 1)):
                    result_row[c] = color

    def do_extrapolation(grid, section_size, num_sections, bg):
        """Extrapolate patterns from sections"""
        cols = len(grid[0])

        # Extract sections
        sections = [grid[i*section_size:(i+1)*section_size] for i in range(num_sections)]

        # Build result
        result = [[bg] * cols for _ in range(section_size)]

        for row_idx in range(section_size):
            # Find contiguous color spans in each section for this row
            section_spans = []
            for sec in sections:
                spans = find_color_spans(sec[row_idx], bg)
                section_spans.append(spans)

            # Match and extrapolate spans
            extrapolate_row(section_spans, result[row_idx], bg)

        return result

    def has_matching_frames(grid, section_size, num_sections):
        """Check if all sections have the same frame rows"""
        frame_rows = min(2, section_size // 4)
        if frame_rows == 0:
            return True

        first_top = [tuple(grid[i]) for i in range(frame_rows)]
        first_bottom = [tuple(grid[i]) for i in range(section_size - frame_rows, section_size)]

        for sec_idx in range(1, num_sections):
            start = sec_idx * section_size
            top = [tuple(grid[start + i]) for i in range(frame_rows)]
            bottom = [tuple(grid[start + i]) for i in range(section_size - frame_rows, section_size)]

            if top != first_top or bottom != first_bottom:
                return False

        return True

    def extrapolate_sections(grid):
        if not grid or not grid[0]:
            return None

        rows, cols = len(grid), len(grid[0])

        # Find background color (most frequent)
        from collections import Counter
        bg = Counter(val for row in grid for val in row).most_common(1)[0][0]

        # Try different section sizes
        for section_size in range(5, rows // 2 + 1):
            if rows % section_size != 0:
                continue

            num_sections = rows // section_size
            if num_sections < 2:
                continue

            # Check if sections have matching frame rows
            if not has_matching_frames(grid, section_size, num_sections):
                continue

            # Found valid sections, extrapolate
            return do_extrapolation(grid, section_size, num_sections, bg)

        return None

    def solve_directional(grid, transpose):
        if transpose:
            grid = [list(row) for row in zip(*grid)]

        result = extrapolate_sections(grid)

        if result is not None and transpose:
            result = [list(row) for row in zip(*result)]

        return result

    # Try vertical sections (rows stacked)
    result = solve_directional(grid, transpose=False)
    if result is not None:
        return result

    # Try horizontal sections (columns side-by-side)
    result = solve_directional(grid, transpose=True)
    if result is not None:
        return result

    return grid
