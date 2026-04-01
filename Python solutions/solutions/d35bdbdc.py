def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains 3×3 hollow frames (8 border cells of same color + 1 center cell)
    2. Build a directed graph where frame A → frame B if A's border color matches B's center value
    3. Find all chains in this graph (starting from nodes with in-degree 0)
    4. Keep frames at even distance from chain end (distance 0, 2, 4...)
    5. Each kept frame's center becomes the center value of its predecessor in chain
    6. Remove all non-kept frames (set all their cells to 0)

    Procedure:
    1. Identify all 3×3 hollow frames with their border color and center value
    2. Build directed graph based on border-color to center-value matching
    3. Find all chains starting from frames with no incoming edges
    4. For each chain, determine which frames to keep (even distance from end)
    5. Update centers of kept frames with predecessor's center value
    6. Clear all removed frames from the grid
    """

    rows, cols = len(grid), len(grid[0])

    # Step 1: Identify all 3×3 hollow frames
    frames = []
    frame_positions = {}  # color -> (r, c, center_value)

    for r in range(rows - 2):
        for c in range(cols - 2):
            # Check if this is a 3×3 frame (all 8 border cells have same color)
            border_color = grid[r][c]
            if border_color == 0:
                continue

            border_cells = [
                grid[r][c], grid[r][c+1], grid[r][c+2],
                grid[r+1][c], grid[r+1][c+2],
                grid[r+2][c], grid[r+2][c+1], grid[r+2][c+2]
            ]

            if len(set(border_cells)) == 1 and border_cells[0] == border_color:
                center_value = grid[r+1][c+1]
                # Only add if this color hasn't been seen before
                if border_color not in frame_positions:
                    frame_positions[border_color] = (r, c, center_value)
                    frames.append((border_color, r, c, center_value))

    # Step 2: Build directed graph (A -> B if A's color matches B's center)
    graph = {}
    for color, r, c, center_value in frames:
        # Find if there's a frame whose center equals this color
        for other_color, other_r, other_c, other_center in frames:
            if other_center == color:
                graph[color] = other_color
                break

    # Step 3: Find all chains starting from nodes with in-degree 0
    in_degree = {color: 0 for color, _, _, _ in frames}
    for src in graph:
        if graph[src] in in_degree:
            in_degree[graph[src]] += 1

    chains = []
    for color, _, _, _ in frames:
        if in_degree[color] == 0:
            chain = [color]
            current = color
            while current in graph:
                current = graph[current]
                chain.append(current)
            chains.append(chain)

    # Step 4: Determine which frames to keep and their new centers
    frames_to_keep = set()
    frame_new_centers = {}

    for chain in chains:
        n = len(chain)
        for i in range(n):
            keep_frame = False
            if n % 2 == 0:
                # Even length chain: keep odd positions (1, 3, 5, ...)
                if i % 2 == 1:
                    keep_frame = True
            else:
                # Odd length chain: keep even positions >= 2 (2, 4, 6, ...)
                if i % 2 == 0 and i >= 2:
                    keep_frame = True

            if keep_frame:
                frames_to_keep.add(chain[i])
                # Update center with predecessor's center value
                prev_color = chain[i - 1]
                r, c, prev_center = frame_positions[prev_color]
                frame_new_centers[chain[i]] = prev_center

    # Step 5: Create output grid
    result = [row[:] for row in grid]

    # Clear removed frames (set all their cells to 0)
    for color, r, c, center_value in frames:
        if color not in frames_to_keep:
            for rr in range(r, r + 3):
                for cc in range(c, c + 3):
                    result[rr][cc] = 0

    # Update centers of kept frames with new values
    for color in frame_new_centers:
        r, c, _ = frame_positions[color]
        result[r + 1][c + 1] = frame_new_centers[color]

    return result
