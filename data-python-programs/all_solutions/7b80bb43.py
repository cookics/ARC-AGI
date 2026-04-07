def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background color and scattered line segments
    2. Output extends line segments by filling small gaps between them
    3. For rows with long segments (>=9), fill gaps <=3
    4. For columns with significant presence (total>=10), fill gaps <=2
    5. Scattered cells not part of major structures are removed

    Procedure:
    1. Identify background and line colors
    2. For each row, fill small gaps if there's a long segment
    3. For each column, fill small gaps if there's enough total presence
    4. Keep only segments that are part of significant structures
    """
    from collections import Counter

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Identify colors
    all_cells = [cell for row in grid for cell in row]
    color_counts = Counter(all_cells)
    sorted_colors = color_counts.most_common()
    background = sorted_colors[0][0]
    line_color = sorted_colors[1][0] if len(sorted_colors) > 1 else background

    def get_segments(line):
        """Get (start, end) tuples for non-background segments"""
        segments = []
        start = None
        for i, val in enumerate(line):
            if val == line_color:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i - 1))
                    start = None
        if start is not None:
            segments.append((start, len(line) - 1))
        return segments

    # Process rows - fill gaps in rows with long segments
    for r in range(rows):
        segments = get_segments(result[r])
        if len(segments) < 2:
            continue

        max_len = max(s[1] - s[0] + 1 for s in segments)

        if max_len >= 9:
            # Fill all small gaps
            for i in range(len(segments) - 1):
                s1_start, s1_end = segments[i]
                s2_start, s2_end = segments[i + 1]
                gap = s2_start - s1_end - 1
                len1 = s1_end - s1_start + 1

                # Only fill gaps adjacent to long segments
                if gap <= 3 and gap > 0 and len1 >= 9:
                    for c in range(s1_end + 1, s2_start):
                        result[r][c] = line_color

    # Process columns - fill gaps in columns with significant presence
    for c in range(cols):
        col_line = [result[r][c] for r in range(rows)]
        segments = get_segments(col_line)
        if len(segments) < 2:
            continue

        total_cells = sum(s[1] - s[0] + 1 for s in segments)
        max_len = max(s[1] - s[0] + 1 for s in segments)

        # Fill gaps if column is significant
        if total_cells >= 10:
            for i in range(len(segments) - 1):
                s1_start, s1_end = segments[i]
                s2_start, s2_end = segments[i + 1]
                gap = s2_start - s1_end - 1

                if gap <= 2 and gap > 0:
                    for r in range(s1_end + 1, s2_start):
                        result[r][c] = line_color

    # Cleanup - keep only significant segments
    keep = [[False] * cols for _ in range(rows)]

    # Mark horizontal segments to keep
    for r in range(rows):
        segments = get_segments(result[r])
        if not segments:
            continue
        total = sum(s[1] - s[0] + 1 for s in segments)
        max_seg = max(s[1] - s[0] + 1 for s in segments)

        if total >= 15 or max_seg >= 9:
            # Keep all segments
            for s_start, s_end in segments:
                for c in range(s_start, s_end + 1):
                    keep[r][c] = True
        else:
            # Keep only segments >= 3
            for s_start, s_end in segments:
                if s_end - s_start + 1 >= 3:
                    for c in range(s_start, s_end + 1):
                        keep[r][c] = True

    # Mark vertical segments to keep
    for c in range(cols):
        col_line = [result[r][c] for r in range(rows)]
        segments = get_segments(col_line)
        if not segments:
            continue
        total = sum(s[1] - s[0] + 1 for s in segments)
        max_seg = max(s[1] - s[0] + 1 for s in segments)

        if total >= 10:
            # Keep all segments
            for s_start, s_end in segments:
                for r in range(s_start, s_end + 1):
                    keep[r][c] = True
        else:
            # Keep segments >= 3
            for s_start, s_end in segments:
                if s_end - s_start + 1 >= 3:
                    for r in range(s_start, s_end + 1):
                        keep[r][c] = True

    # Remove unmarked cells
    for r in range(rows):
        for c in range(cols):
            if result[r][c] == line_color and not keep[r][c]:
                result[r][c] = background

    return result
