def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background value 8
    2. There's a 9×9 rectangular pattern with non-8 values
    3. The pattern has a 3×3 hollow center filled with 8s
    4. Output fills the hollow center based on surrounding values
    5. Output also reflects/extends the pattern into 3×3 areas around the pattern

    Procedure:
    1. Find the 9×9 pattern (bounding box of non-8 values)
    2. Identify the 3×3 hollow center within the pattern
    3. Fill the hollow center using values from the surrounding 5×5 frame:
       - Corners: from 5×5 frame corners (diagonal neighbors)
       - Edge midpoints: from 5×5 frame edge midpoints
       - Center: from non-8 values at specific positions in 5×5 frame
    4. Reflect pattern outward into surrounding 3×3 areas
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Find bounding box of non-8 values (the 9×9 pattern)
    min_r = max_r = min_c = max_c = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 8:
                if min_r is None:
                    min_r = r
                max_r = r
                if min_c is None or c < min_c:
                    min_c = c
                if max_c is None or c > max_c:
                    max_c = c

    # Extract the 9×9 pattern
    pattern = []
    for r in range(min_r, max_r + 1):
        pattern.append(grid[r][min_c:max_c + 1])

    # The hollow center should be at pattern[3:6, 3:6] for a 9×9 pattern
    # Fill the hollow center using the 5×5 frame (pattern[2:7, 2:7])

    # Extract 5×5 frame around the hollow
    frame = []
    for r in range(2, 7):
        frame.append(pattern[r][2:7])

    # Fill the 3×3 hollow
    filled = [[0]*3 for _ in range(3)]

    # Corners: from 5×5 frame corners (diagonal neighbors of hollow)
    filled[0][0] = frame[0][0]
    filled[0][2] = frame[0][4]
    filled[2][0] = frame[4][0]
    filled[2][2] = frame[4][4]

    # Edge midpoints: from 5×5 frame edge midpoints
    filled[0][1] = frame[0][2]
    filled[1][0] = frame[2][0]
    filled[1][2] = frame[2][4]
    filled[2][1] = frame[4][2]

    # Center: find a non-8 value from the frame that's different from corners/edges
    # Collect all unique non-8 values from the 5×5 frame
    frame_values = set()
    for r in range(5):
        for c in range(5):
            if frame[r][c] != 8:
                frame_values.add(frame[r][c])

    # Try to find a value different from both corner and edge values
    corner_val = filled[0][0]
    edge_val = filled[0][1]
    for val in frame_values:
        if val != corner_val and val != edge_val:
            filled[1][1] = val
            break
    else:
        # If no such value, try different from corner only
        for val in frame_values:
            if val != corner_val:
                filled[1][1] = val
                break
        else:
            # If all values are the same, use that value
            filled[1][1] = corner_val if corner_val != 0 else (list(frame_values)[0] if frame_values else 0)

    # Write filled center back to result and pattern
    for r in range(3):
        for c in range(3):
            result[min_r + 3 + r][min_c + 3 + c] = filled[r][c]
            pattern[3 + r][3 + c] = filled[r][c]

    # Now reflect/extend the pattern into the background
    # Divide into 9 regions - extend corners and edges

    # Helper function to safely set values
    def set_region(result, out_r, out_c, pat_r, pat_c):
        if 0 <= out_r < rows and 0 <= out_c < cols:
            if 0 <= pat_r < 9 and 0 <= pat_c < 9:
                result[out_r][out_c] = pattern[pat_r][pat_c]

    # Top-left background: gets bottom-right corner of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = min_r - 3 + dr
            out_c = min_c - 3 + dc
            pat_r = 6 + dr
            pat_c = 6 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Top-middle background: gets bottom-middle of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = min_r - 3 + dr
            out_c = min_c + 3 + dc
            pat_r = 6 + dr
            pat_c = 3 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Top-right background: gets bottom-left corner of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = min_r - 3 + dr
            out_c = max_c + 1 + dc
            pat_r = 6 + dr
            pat_c = 0 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Middle-left background: gets middle-right of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = min_r + 3 + dr
            out_c = min_c - 3 + dc
            pat_r = 3 + dr
            pat_c = 6 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Middle-right background: gets middle-left of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = min_r + 3 + dr
            out_c = max_c + 1 + dc
            pat_r = 3 + dr
            pat_c = 0 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Bottom-left background: gets top-right corner of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = max_r + 1 + dr
            out_c = min_c - 3 + dc
            pat_r = 0 + dr
            pat_c = 6 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Bottom-middle background: gets top-middle of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = max_r + 1 + dr
            out_c = min_c + 3 + dc
            pat_r = 0 + dr
            pat_c = 3 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    # Bottom-right background: gets top-left corner of pattern
    for dr in range(3):
        for dc in range(3):
            out_r = max_r + 1 + dr
            out_c = max_c + 1 + dc
            pat_r = 0 + dr
            pat_c = 0 + dc
            set_region(result, out_r, out_c, pat_r, pat_c)

    return result
