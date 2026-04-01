def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input is a 2D grid of integers with dimensions H x W
    2. Output is a 2D grid with dimensions 2*H x 2*W (exactly double both dimensions)
    3. The transformation creates a 4-way mirrored version of the input
    4. Each row is first horizontally mirrored by concatenating original row with its reverse
    5. The entire horizontally mirrored grid is then vertically mirrored by concatenating it with its reverse

    Procedure:
    1. For each row in input, create horizontally mirrored row (original + horizontal flip)
    2. Create top half by collecting all horizontally mirrored rows
    3. Create bottom half by vertically flipping the top half
    4. Concatenate top half and bottom half
    """

    # Step 1: Create horizontally mirrored rows
    top_half = []
    for row in grid:
        # Concatenate original row with its horizontal flip
        mirrored_row = row + row[::-1]
        top_half.append(mirrored_row)

    # Step 2: Create bottom half by vertically flipping top half
    bottom_half = top_half[::-1]

    # Step 3: Combine top and bottom halves
    result = top_half + bottom_half

    return result
