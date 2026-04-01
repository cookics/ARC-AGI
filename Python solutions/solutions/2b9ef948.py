def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains two distinct patterns:
       a) A 3x3 box with border color 4 and center color X
       b) A path made of 5s with two special endpoints (fill color and 4)
    2. Output is filled with the fill color from the path endpoint
    3. Two diagonal lines (slopes +1 and -1) cross forming an X pattern
    4. The 3x3 box is placed at the crossing point
    5. The diagonals extend to connect/encompass both input patterns
    6. Output dimensions are calculated to fit the diagonal extent

    Procedure:
    1. Find the 3x3 box and extract center color
    2. Find the 5-path and determine fill color
    3. Calculate where diagonals should cross based on pattern geometry
    4. Compute output dimensions based on diagonal extent
    5. Fill output with fill color, draw diagonals, place box at crossing
    """

    H, W = len(grid), len(grid[0])

    # Find the 3x3 box
    box_center = None
    box_center_color = None
    for r in range(H-2):
        for c in range(W-2):
            if (grid[r][c] == 4 and grid[r][c+1] == 4 and grid[r][c+2] == 4 and
                grid[r+2][c] == 4 and grid[r+2][c+1] == 4 and grid[r+2][c+2] == 4 and
                grid[r+1][c] == 4 and grid[r+1][c+2] == 4):
                box_center = (r+1, c+1)
                box_center_color = grid[r+1][c+1]
                break
        if box_center:
            break

    # Find all cells with value 5 (the path)
    fives = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 5:
                fives.append((r, c))

    # Find bounding box of 5-path
    if fives:
        path_min_r = min(r for r, c in fives)
        path_max_r = max(r for r, c in fives)
        path_min_c = min(c for r, c in fives)
        path_max_c = max(c for r, c in fives)

    # Find path endpoints (non-5, non-0 cells adjacent to 5s, excluding box cells)
    path_endpoints = []
    box_r, box_c = box_center
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0 and grid[r][c] != 5:
                # Skip if it's part of the 3x3 box
                if abs(r - box_r) <= 1 and abs(c - box_c) <= 1:
                    continue
                # Check if adjacent to a 5
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 5:
                        path_endpoints.append((r, c, grid[r][c]))
                        break

    # Determine fill color and get fill endpoint
    fill_color = None
    fill_endpoint = None
    for r, c, v in path_endpoints:
        if v != 4:
            fill_color = v
            fill_endpoint = (r, c)
            break

    # Include endpoints in path bounding box
    all_path_cells = fives + [(r, c) for r, c, v in path_endpoints]
    if all_path_cells:
        path_min_r = min(r for r, c in all_path_cells)
        path_max_r = max(r for r, c in all_path_cells)
        path_min_c = min(c for r, c in all_path_cells)
        path_max_c = max(c for r, c in all_path_cells)

    # Use fill endpoint as reference point for path
    pr, pc = fill_endpoint
    br, bc = box_center

    # Check if path is vertical (single column)
    path_is_vertical = (path_min_c == path_max_c)

    # Calculate crossing point
    if path_is_vertical and box_r < path_min_r:
        # Special case: vertical path below box
        # Crossing is between them, ratio based on their separation
        cross_r = br + (path_min_r - br) * (5.0 / 7.0)
        cross_c = bc
    else:
        # General case: use diagonal intersection formula
        # Intersection of r - c = br - bc and r + c = pr + pc
        base_cross_r = (br - bc + pr + pc) / 2
        base_cross_c = (pr + pc - br + bc) / 2

        # Apply adjustment based on pattern position
        if box_r > path_max_r:  # Box below path
            cross_r = base_cross_r + 2
            cross_c = base_cross_c
        elif box_r < path_min_r:  # Box above path
            cross_r = base_cross_r + 1.5
            cross_c = base_cross_c + 1.5
        else:  # Overlapping
            cross_r = base_cross_r + 1
            cross_c = base_cross_c

    # Calculate diagonal constants
    const1 = cross_r + cross_c  # for slope -1 (r + c = constant)
    const2 = cross_r - cross_c  # for slope +1 (r - c = constant)

    # Determine output dimensions
    # For square grids, keep the same dimensions
    # For non-square, calculate based on diagonal extent
    if H == W:
        out_h = H
    else:
        # Calculate the rows needed for diagonals to reach edges
        out_h = int(cross_r) + max(int(cross_c), int(W - 1 - cross_c)) + 3
        out_h = min(out_h, H)  # Don't exceed input height

    out_w = W

    # Create output grid
    result = [[fill_color] * out_w for _ in range(out_h)]

    # Draw diagonal 1 (slope -1): r + c = const1
    for r in range(out_h):
        c = int(const1 - r)
        if 0 <= c < out_w:
            result[r][c] = 4

    # Draw diagonal 2 (slope +1): r - c = const2
    for r in range(out_h):
        c = int(r - const2)
        if 0 <= c < out_w:
            result[r][c] = 4

    # Place 3x3 box at crossing point (round to nearest integer)
    center_r = int(cross_r + 0.5)
    center_c = int(cross_c + 0.5)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r, c = center_r + dr, center_c + dc
            if 0 <= r < out_h and 0 <= c < out_w:
                if dr == 0 and dc == 0:
                    result[r][c] = box_center_color
                else:
                    result[r][c] = 4

    return result
