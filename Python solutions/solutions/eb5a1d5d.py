def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains nested rectangular regions with different values
    2. Output is a schematic representation where each layer is shown as a frame
    3. Layers are identified by sampling along the diagonal from corner to center
    4. A value can appear at multiple nesting levels (e.g., background and inner)
    5. Output size = 2*n - 1, where n is the number of distinct layers

    Procedure:
    1. Sample values along diagonal from (0,0) towards center
    2. Record distinct layers (ignoring consecutive duplicates)
    3. Create output grid of size (2*n - 1) x (2*n - 1)
    4. Fill each layer as a frame at the appropriate distance from edge
    """

    rows = len(grid)
    cols = len(grid[0])

    # Sample along diagonal from top-left to identify nesting layers
    # Stop when we encounter a value we've already seen (exiting a nested region)
    layers = []
    seen_values = set()

    for step in range(min(rows, cols)):
        val = grid[step][step]
        # Only process if different from previous layer
        if not layers or layers[-1] != val:
            # Stop if we've seen this value before (we're exiting)
            if val in seen_values:
                break
            layers.append(val)
            seen_values.add(val)

    # Create output grid
    n = len(layers)
    size = 2 * n - 1
    result = [[0] * size for _ in range(size)]

    # Fill from outside to inside
    for i in range(n):
        val = layers[i]
        for r in range(size):
            for c in range(size):
                # Fill cells at distance i from edge
                min_dist = min(r, c, size - 1 - r, size - 1 - c)
                if min_dist == i:
                    result[r][c] = val

    return result
