def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input grid contains rectangular frames made of specific values (1,2,3,4,8, etc.)
    2. There's a "noise" value scattered throughout (5, 7, or 9)
    3. Each distinct non-zero, non-noise value represents one frame
    4. Output has one row per frame: [frame_val repeated N times, noise repeated (4-N) times]
    5. N ranges from 1 to 4, encoding frame properties (cleanliness/size)
    6. Frames are sorted by spatial position + size properties

    Procedure:
    1. Identify noise value (one of 5, 7, 9)
    2. Find all distinct frame values
    3. For each frame, compute bounding box
    4. Compute metrics: min_row, min_col, area
    5. Assign repetitions (1-4) based on frame properties
    6. Sort frames and generate output
    """

    rows, cols = len(grid), len(grid[0])

    # Find noise value
    noise_val = None
    for candidate in [5, 7, 9]:
        if any(candidate in row for row in grid):
            noise_val = candidate
            break
    if noise_val is None:
        noise_val = 5

    # Find all distinct frame values (non-zero, non-noise)
    frame_values = set()
    for row in grid:
        for val in row:
            if val != 0 and val != noise_val:
                frame_values.add(val)

    # For each frame value, find its bounding box and properties
    frame_info = []
    for frame_val in frame_values:
        # Find bounding box
        min_r, max_r = rows, -1
        min_c, max_c = cols, -1
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == frame_val:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

        width = max_c - min_c + 1
        height = max_r - min_r + 1
        area = width * height
        perimeter = 2 * (width + height)
        aspect_ratio = width / height if height > 0 else 0

        # Count how many frame cells and noise cells in bbox
        frame_count = 0
        noise_count = 0
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r][c] == frame_val:
                    frame_count += 1
                elif grid[r][c] == noise_val:
                    noise_count += 1

        frame_info.append({
            'val': frame_val,
            'min_r': min_r,
            'min_c': min_c,
            'max_r': max_r,
            'max_c': max_c,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'frame_count': frame_count,
            'noise_count': noise_count
        })

    # Assign repetitions based on aspect ratio (with perimeter as tiebreaker)
    sorted_by_size = sorted(frame_info, key=lambda x: (x['aspect_ratio'], x['perimeter']))
    n = len(sorted_by_size)

    # Generate repetition pattern based on number of frames
    if n <= 2:
        rep_pattern = list(range(1, n + 1))
    elif n == 3:
        rep_pattern = [1, 2, 4]
    else:  # n >= 4
        rep_pattern = [1, 2] + [3] * (n - 3) + [4]

    val_to_reps = {}
    for idx, info in enumerate(sorted_by_size):
        val_to_reps[info['val']] = rep_pattern[idx]

    # Add repetitions to frame_info
    for info in frame_info:
        info['reps'] = val_to_reps[info['val']]

    # Sort frames by aspect ratio for output order (same as rep assignment)
    frame_info.sort(key=lambda x: (x['aspect_ratio'], x['perimeter']))

    # Generate output
    result = []
    for info in frame_info:
        reps = info['reps']
        row = [info['val']] * reps + [noise_val] * (4 - reps)
        result.append(row)

    return result
