def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a grid with 0s (background) and 5s forming rectangular regions
    2. Output is a 9x9 grid showing a transformed pattern
    3. The transformation involves:
       - Finding the bounding box of all 5s (should be 9x9)
       - Dividing that region into 3x3 blocks
       - Determining which blocks are filled with 5s
       - Filling each block with a specific pattern based on the overall configuration

    Procedure:
    1. Find bounding box of all cells with value 5
    2. Extract the 9x9 region
    3. Divide into 3x3 blocks and identify filled blocks
    4. Determine pattern type based on connectivity:
       - All blocks isolated → Pattern B (X pattern)
       - Largest component size ≤ 2 → Pattern A
       - Largest component size > 2 → Pattern C
    5. Fill output with the appropriate pattern for each filled block
    """

    # Find bounding box of all 5s
    rows_with_5 = [i for i in range(len(grid)) if any(cell == 5 for cell in grid[i])]
    cols_with_5 = [j for j in range(len(grid[0])) if any(grid[i][j] == 5 for i in range(len(grid)))]

    if not rows_with_5 or not cols_with_5:
        return [[0] * 9 for _ in range(9)]

    min_row, max_row = min(rows_with_5), max(rows_with_5)
    min_col, max_col = min(cols_with_5), max(cols_with_5)

    # Extract region
    extracted = [grid[i][min_col:max_col+1] for i in range(min_row, max_row+1)]

    # Determine which 3x3 blocks are completely filled with 5s
    height, width = len(extracted), len(extracted[0])
    block_rows, block_cols = height // 3, width // 3

    filled_blocks = set()
    for br in range(block_rows):
        for bc in range(block_cols):
            if all(extracted[br*3+i][bc*3+j] == 5 for i in range(3) for j in range(3)):
                filled_blocks.add((br, bc))

    # Find connected components to determine pattern type
    def find_components(blocks):
        visited = set()
        components = []

        for br, bc in blocks:
            if (br, bc) in visited:
                continue
            component = set()
            queue = [(br, bc)]
            while queue:
                r, c = queue.pop(0)
                if (r, c) in visited:
                    continue
                visited.add((r, c))
                component.add((r, c))
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) in blocks and (nr, nc) not in visited:
                        queue.append((nr, nc))
            components.append(component)

        return components

    components = find_components(filled_blocks)
    max_component_size = max(len(comp) for comp in components) if components else 0

    # Determine pattern type
    if max_component_size == 1:  # All isolated
        pattern = [
            [5, 0, 5],
            [0, 5, 0],
            [5, 0, 5]
        ]
    elif max_component_size == 2:  # Small components
        pattern = [
            [5, 5, 0],
            [0, 0, 5],
            [5, 5, 0]
        ]
    else:  # Large connected component
        pattern = [
            [5, 5, 5],
            [0, 5, 5],
            [5, 0, 5]
        ]

    # Create output
    output = [[0] * (block_cols * 3) for _ in range(block_rows * 3)]

    # Fill patterns for each filled block
    for br, bc in filled_blocks:
        for i in range(3):
            for j in range(3):
                output[br*3+i][bc*3+j] = pattern[i][j]

    return output
