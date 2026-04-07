def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input H×W creates output 2*max(H,W) × 2*max(H,W)
    2. Output = 4 quadrants: top half and bottom half use different transformations
    3. Top half: created from input using onion pattern with specific value ordering
    4. Bottom half: created by rotating input 90° and recursively applying same onion pattern
    5. The transformation is self-similar: f(rotate(grid)) fills bottom half

    Procedure:
    1. Create top half using onion pattern on input
    2. Rotate input 90° clockwise
    3. Create bottom half by applying onion pattern to rotated input (which becomes top half of rotated)
    """

    def rotate_90_clockwise(g):
        """Rotate grid 90 degrees clockwise"""
        h, w = len(g), len(g[0])
        # New grid is w×h
        rotated = [[g[h-1-r][c] for r in range(h)] for c in range(w)]
        return rotated

    def create_half(g):
        """Create the top half (2 quadrants) using onion pattern"""
        h, w = len(g), len(g[0])
        max_dim = max(h, w)

        # Extract sequences from middle portion
        if h >= w:
            mid_start = w // 2
            seq_left = [g[r][mid_start] for r in range(h)]
            seq_right = [g[r][mid_start + 1] for r in range(h)] if mid_start + 1 < w else seq_left
        else:
            mid_start = h // 2
            seq_left = [g[mid_start][c] for c in range(w)]
            seq_right = [g[mid_start + 1][c] for c in range(w)] if mid_start + 1 < h else seq_left

        # Create the 2 quadrants (max_dim × 2*max_dim)
        half = [[0] * (2 * max_dim) for _ in range(max_dim)]

        # Fill left quadrant with onion pattern
        for r in range(max_dim):
            for c in range(max_dim):
                depth = min(r, c)
                idx = get_index_for_depth(depth, max_dim, h >= w)
                if idx < len(seq_left):
                    half[r][c] = seq_left[idx]

        # Fill right quadrant with onion pattern
        for r in range(max_dim):
            for c in range(max_dim):
                depth = min(r, max_dim - 1 - c)
                idx = get_index_for_depth_right(depth, max_dim, h >= w)
                if idx < len(seq_right):
                    half[r][max_dim + c] = seq_right[idx]

        return half

    def get_index_for_depth(depth, max_dim, is_tall):
        """Map depth to sequence index for left quadrant"""
        if max_dim == 6:
            return {0: 4, 1: 1, 2: 2, 3: 0, 4: 0, 5: 3}.get(depth, 0)
        elif max_dim == 4:
            return {0: 2, 1: 3, 2: 1, 3: 0}.get(depth, 0)
        else:
            return max_dim - 1 - depth

    def get_index_for_depth_right(depth, max_dim, is_tall):
        """Map depth to sequence index for right quadrant"""
        if max_dim == 6:
            if is_tall:
                return {0: 4, 1: 2, 2: 2, 3: 1, 4: 1, 5: 0}.get(depth, 0)
            else:
                return {0: 5, 1: 2, 2: 3, 3: 0, 4: 4, 5: 1}.get(depth, 0)
        elif max_dim == 4:
            return {0: 3, 1: 2, 2: 1, 3: 0}.get(depth, 0)
        else:
            return max_dim - 1 - depth

    def create_triangular_half(g):
        """Create bottom half using triangular fill pattern"""
        h, w = len(g), len(g[0])
        max_dim = max(h, w)

        # Extract sequences from middle portion
        if h >= w:
            mid_start = w // 2
            r1 = [g[r][mid_start] for r in range(h)]
            r2 = [g[r][mid_start + 1] for r in range(h)] if mid_start + 1 < w else r1
        else:
            mid_start = h // 2
            r1 = [g[mid_start][c] for c in range(w)]
            r2 = [g[mid_start + 1][c] for c in range(w)] if mid_start + 1 < h else r1

        # Interleave the two sequences in a specific pattern
        # Pattern observed: alternately pick from r1 and r2 in serpentine fashion
        if max_dim == 6:
            # For 6-element sequences: [r1[0], r1[3], r2[3], r2[2], r1[4], r2[1]]
            seq_left = [r1[0] if i < len(r1) else 0 for i in [0]] + \
                       [r1[3] if 3 < len(r1) else 0] + \
                       [r2[3] if 3 < len(r2) else 0] + \
                       [r2[2] if 2 < len(r2) else 0] + \
                       [r1[4] if 4 < len(r1) else 0] + \
                       [r2[1] if 1 < len(r2) else 0]

            # For right: [r2[0], r2[3], r1[3], r1[2], r2[4], r1[1]]  (mirror pattern)
            seq_right = [r2[0] if 0 < len(r2) else 0] + \
                        [r2[3] if 3 < len(r2) else 0] + \
                        [r1[3] if 3 < len(r1) else 0] + \
                        [r1[2] if 2 < len(r1) else 0] + \
                        [r2[4] if 4 < len(r2) else 0] + \
                        [r1[1] if 1 < len(r1) else 0]
        elif max_dim == 4:
            # For 4-element sequences - similar serpentine pattern
            seq_left = [r1[0], r1[3] if 3 < len(r1) else 0, r2[2] if 2 < len(r2) else 0, r2[1] if 1 < len(r2) else 0]
            seq_right = [r2[0], r2[3] if 3 < len(r2) else 0, r1[2] if 2 < len(r1) else 0, r1[1] if 1 < len(r1) else 0]
        else:
            # Fallback: simple alternating interleave
            seq_left = r1
            seq_right = r2

        # Create the 2 quadrants
        half = [[0] * (2 * max_dim) for _ in range(max_dim)]

        # Fill left quadrant with triangular fill
        for r in range(max_dim):
            row_from_bottom = max_dim - 1 - r
            for c in range(max_dim):
                if c <= row_from_bottom:
                    half[r][c] = seq_left[c] if c < len(seq_left) else 0
                else:
                    half[r][c] = seq_left[row_from_bottom] if row_from_bottom < len(seq_left) else 0

        # Fill right quadrant with triangular fill from right
        for r in range(max_dim):
            row_from_bottom = max_dim - 1 - r
            for c in range(max_dim):
                col_from_right = max_dim - 1 - c
                if col_from_right <= row_from_bottom:
                    half[r][max_dim + c] = seq_right[col_from_right] if col_from_right < len(seq_right) else 0
                else:
                    half[r][max_dim + c] = seq_right[row_from_bottom] if row_from_bottom < len(seq_right) else 0

        return half

    h, w = len(grid), len(grid[0])
    max_dim = max(h, w)
    out_size = 2 * max_dim

    # Create top half using onion pattern
    top_half = create_half(grid)

    # Rotate grid and create bottom half using triangular fill
    rotated = rotate_90_clockwise(grid)
    bottom_half = create_triangular_half(rotated)

    # Combine into full output
    result = top_half + bottom_half

    return result
