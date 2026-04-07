from collections import defaultdict


def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    Keep all pixels of the most frequent color. Among remaining colors,
    choose the most frequent one to create a triangular fill from the corner
    farthest from the main color's center of mass.

    Procedure:
    1. Count frequency of each color
    2. Keep the most frequent color (main color)
    3. Choose second most frequent color for fill
    4. Fill triangle from corner farthest from main color center
    """

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Count frequency of each color
    color_count = defaultdict(int)
    color_positions = defaultdict(list)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color = grid[r][c]
                color_count[color] += 1
                color_positions[color].append((r, c))

    if not color_count:
        return result

    # Find the most frequent color (main color to keep)
    main_color = max(color_count.keys(), key=lambda x: color_count[x])

    # Place all pixels of the main color
    main_positions = set()
    for r, c in color_positions[main_color]:
        result[r][c] = main_color
        main_positions.add((r, c))

    # Find the second most frequent color for fill
    remaining_colors = {
        color: count for color, count in color_count.items() if color != main_color
    }

    if not remaining_colors:
        return result

    fill_color = max(remaining_colors.keys(), key=lambda x: remaining_colors[x])

    # Test all corners and choose the one that allows largest triangle
    def test_triangle_size(corner_r, corner_c, corner_name):
        fill_positions = []

        if corner_name == "top-left":
            for size in range(1, max(rows, cols)):
                valid_positions = []
                for i in range(size):
                    r, c = i, size - 1 - i
                    if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                        valid_positions.append((r, c))
                    else:
                        break

                if len(valid_positions) == size:
                    fill_positions.extend(valid_positions)
                else:
                    break

        elif corner_name == "bottom-left":
            for size in range(1, max(rows, cols)):
                valid_positions = []
                for i in range(size):
                    r, c = rows - 1 - i, size - 1 - i
                    if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                        valid_positions.append((r, c))
                    else:
                        break

                if len(valid_positions) == size:
                    fill_positions.extend(valid_positions)
                else:
                    break

        elif corner_name == "bottom-right":
            for size in range(1, max(rows, cols)):
                valid_positions = []
                for i in range(size):
                    r, c = rows - 1 - i, cols - 1 - (size - 1 - i)
                    if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                        valid_positions.append((r, c))
                    else:
                        break

                if len(valid_positions) == size:
                    fill_positions.extend(valid_positions)
                else:
                    break

        elif corner_name == "top-right":
            for size in range(1, max(rows, cols)):
                valid_positions = []
                for i in range(size):
                    r, c = i, cols - 1 - (size - 1 - i)
                    if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                        valid_positions.append((r, c))
                    else:
                        break

                if len(valid_positions) == size:
                    fill_positions.extend(valid_positions)
                else:
                    break

        return len(fill_positions)

    # Test all corners
    corners = [
        (0, 0, "top-left"),
        (0, cols - 1, "top-right"),
        (rows - 1, 0, "bottom-left"),
        (rows - 1, cols - 1, "bottom-right"),
    ]

    best_corner = max(
        corners, key=lambda corner: test_triangle_size(corner[0], corner[1], corner[2])
    )

    corner_r, corner_c, corner_name = best_corner

    # Fill triangle from the chosen corner - stop when hitting main color
    if corner_name == "top-left":
        # Fill down and right from (0,0)
        for size in range(1, max(rows, cols)):
            valid_positions = []
            for i in range(size):
                r, c = i, size - 1 - i
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                    valid_positions.append((r, c))
                else:
                    # Hit boundary or main cluster
                    break

            # If this entire layer is valid, place it
            if len(valid_positions) == size:
                for r, c in valid_positions:
                    result[r][c] = fill_color
            else:
                # Partial layer means we hit the main cluster, stop
                break

    elif corner_name == "bottom-left":
        # Fill up and right from (rows-1, 0)
        for size in range(1, max(rows, cols)):
            valid_positions = []
            for i in range(size):
                r, c = rows - 1 - i, size - 1 - i
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                    valid_positions.append((r, c))
                else:
                    break

            if len(valid_positions) == size:
                for r, c in valid_positions:
                    result[r][c] = fill_color
            else:
                break

    elif corner_name == "bottom-right":
        # Fill up and left from (rows-1, cols-1)
        for size in range(1, max(rows, cols)):
            valid_positions = []
            for i in range(size):
                r, c = rows - 1 - i, cols - 1 - (size - 1 - i)
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                    valid_positions.append((r, c))
                else:
                    break

            if len(valid_positions) == size:
                for r, c in valid_positions:
                    result[r][c] = fill_color
            else:
                break

    elif corner_name == "top-right":
        # Fill down and left from (0, cols-1)
        for size in range(1, max(rows, cols)):
            valid_positions = []
            for i in range(size):
                r, c = i, cols - 1 - (size - 1 - i)
                if 0 <= r < rows and 0 <= c < cols and (r, c) not in main_positions:
                    valid_positions.append((r, c))
                else:
                    break

            if len(valid_positions) == size:
                for r, c in valid_positions:
                    result[r][c] = fill_color
            else:
                break

    return result
