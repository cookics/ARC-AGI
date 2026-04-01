def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with background (4s) and various objects
    2. Some objects form shapes made of 1s
    3. Other objects are isolated special numbers (not 1 or 4)
    4. Output moves/duplicates shapes to align with isolated special numbers
    5. Each shape has an associated marker color found inside or nearby

    Procedure:
    1. Find all shapes made of 1s using DFS/BFS
    2. For each shape, find the nearest special number (or one contained within)
    3. Find isolated instances of that same special number
    4. Move/duplicate shapes to align with isolated instances
    5. Return clean grid with only the repositioned shapes
    """

    rows, cols = len(grid), len(grid[0])
    result = [[4 for _ in range(cols)] for _ in range(rows)]

    # Find all connected components of 1s
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    shapes = []

    def dfs(r, c, shape_cells):
        if r < 0 or r >= rows or c < 0 or c >= cols or visited[r][c]:
            return
        if grid[r][c] != 1:
            return
        visited[r][c] = True
        shape_cells.append((r, c))
        # Check 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, shape_cells)

    # Find all shapes made of 1s
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not visited[r][c]:
                shape_cells = []
                dfs(r, c, shape_cells)
                if shape_cells:
                    shapes.append(shape_cells)

    # Find all special numbers (non-1, non-4)
    special_numbers = {}
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 1 and grid[r][c] != 4:
                num = grid[r][c]
                if num not in special_numbers:
                    special_numbers[num] = []
                special_numbers[num].append((r, c))

    # For each shape, find associated special number
    for shape_cells in shapes:
        # Find center of shape
        center_r = sum(r for r, c in shape_cells) // len(shape_cells)
        center_c = sum(c for r, c in shape_cells) // len(shape_cells)

        # Find closest special number to this shape
        min_distance = float("inf")
        associated_num = None
        reference_pos = None

        for num, positions in special_numbers.items():
            for pos_r, pos_c in positions:
                distance = abs(center_r - pos_r) + abs(center_c - pos_c)
                if distance < min_distance:
                    min_distance = distance
                    associated_num = num
                    reference_pos = (pos_r, pos_c)

        if associated_num is None:
            continue

        # Find all positions of this special number
        all_positions = special_numbers.get(associated_num, [])

        # Find other instances (not the reference position)
        other_positions = []
        for pos in all_positions:
            if pos != reference_pos:
                other_positions.append(pos)

        # Calculate offset from reference position to other instances
        for other_r, other_c in other_positions:
            offset_r = other_r - reference_pos[0]
            offset_c = other_c - reference_pos[1]

            # Place the entire shape at the new position
            for shape_r, shape_c in shape_cells:
                new_r = shape_r + offset_r
                new_c = shape_c + offset_c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    result[new_r][new_c] = 1

            # Place the special number at the other position
            if 0 <= other_r < rows and 0 <= other_c < cols:
                result[other_r][other_c] = associated_num

    return result
