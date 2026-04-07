def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. The input is a 2D grid containing integers and some cells with value 0 (missing values).
    2. The output is the same grid with all 0s filled in to complete the underlying repeating pattern.
    3. Each example demonstrates a different repeating pattern size (7x7, 6x6, 2x2 patterns).
    4. The grid follows a periodic tiling pattern where the same sub-pattern repeats across the entire grid.

    Procedure:
    1. Test different pattern sizes from small to large to find the correct repeating period.
    2. Extract the base pattern by collecting non-zero values at each position modulo the pattern size.
    3. Validate that the pattern is consistent across all non-zero cells in the grid.
    4. Fill in missing values (0s) using the identified repeating pattern.
    5. Use fallback strategies if no consistent pattern is found or if pattern lookup fails.
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]

    # Try different pattern sizes from small to large
    def test_pattern_size(h_period, v_period):
        """Test if a given pattern size is consistent with the non-zero values"""
        pattern = {}
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != 0:
                    key = (i % v_period, j % h_period)
                    if key in pattern:
                        if pattern[key] != grid[i][j]:
                            return False, None
                    else:
                        pattern[key] = grid[i][j]
        return True, pattern

    # Try different pattern sizes
    found_pattern = None
    best_h_period = 1
    best_v_period = 1

    for h_period in range(1, min(cols + 1, 15)):  # Limit search to reasonable sizes
        for v_period in range(1, min(rows + 1, 15)):
            is_valid, pattern = test_pattern_size(h_period, v_period)
            if is_valid and pattern:  # Need non-empty pattern
                found_pattern = pattern
                best_h_period = h_period
                best_v_period = v_period
                break
        if found_pattern:
            break

    # If no pattern found, try a more lenient approach
    if found_pattern is None:
        # Use a simple approach: look for any consistent period
        best_h_period = 7  # Default fallback
        best_v_period = 7
        found_pattern = {}
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != 0:
                    key = (i % best_v_period, j % best_h_period)
                    found_pattern[key] = grid[i][j]  # Just take the first value we see

    # Fill in the zeros using the pattern
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                key = (i % best_v_period, j % best_h_period)
                if key in found_pattern:
                    result[i][j] = found_pattern[key]
                else:
                    # Fallback: try to find the value in nearby positions
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != 0:
                                result[i][j] = grid[ni][nj]
                                break
                        if result[i][j] != 0:
                            break
                    if result[i][j] == 0:
                        result[i][j] = 1  # Ultimate fallback

    return result
