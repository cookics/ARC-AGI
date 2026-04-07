def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Row 5 contains all 4s and acts as a divider that remains unchanged.
    2. Non-7 elements from the bottom half (rows 6-10) get moved to the top half (rows 0-4) using value-specific translation vectors.
    3. Translation vectors: Value 6 (-6, 0), Value 2 (-5, +3), Value 8 (-7, +4), Value 1 (-5, -6).
    4. Non-6 values overwrite 6 values when they map to the same position.
    5. The bottom half gets cleared to all 7s.

    Procedure:
    1. Find all non-7 elements in bottom half (rows 6-10)
    2. Apply appropriate translation vector based on element value
    3. Clear bottom half to all 7s
    """

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # copy the grid

    # Define translation vectors for different values
    translations = {6: (-6, 0), 2: (-5, 3), 8: (-7, 4), 1: (-5, -6)}

    # Find all non-7 elements in bottom half (rows 6-10)
    elements = []
    for r in range(6, rows):
        for c in range(cols):
            if result[r][c] != 7:
                elements.append((r, c, result[r][c]))
                result[r][c] = 7  # clear bottom half

    # Apply translations with collision resolution: non-6 values beat 6 values
    for r, c, val in elements:
        if val in translations:
            dr, dc = translations[val]
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < rows and 0 <= new_c < cols:
                # Place if empty, or if current value beats existing value
                current = result[new_r][new_c]
                if current == 7 or (val != 6 and current == 6):
                    result[new_r][new_c] = val

    return result
