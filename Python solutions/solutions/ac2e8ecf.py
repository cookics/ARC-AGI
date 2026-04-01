def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains scattered colored patterns (connected components)
    2. Output rearranges patterns vertically to top and bottom sections
    3. Middle of grid becomes empty
    4. Patterns preserve their horizontal positions and shapes
    5. Patterns that don't overlap horizontally can share the same output rows
    6. Assignment rule: patterns from bottom half of input go to top of output, and vice versa
    7. Patterns of the same color in the same input half are paired and split between output sections

    Procedure:
    1. Find all connected components
    2. Group components by color
    3. For color pairs in same input half: split them (assign to both top and bottom output)
    4. For individual components or cross-half pairs: swap halves (top input → bottom output, vice versa)
    5. Sort and pack components efficiently in output sections
    """

    if not grid or not grid[0]:
        return grid

    rows, cols = len(grid), len(grid[0])
    result = [[0] * cols for _ in range(rows)]

    # Find all connected components
    visited = set()
    shapes = []

    def flood_fill(start_r, start_c, color):
        """Find all cells in a connected component"""
        cells = []
        stack = [(start_r, start_c)]
        component_visited = set()

        while stack:
            r, c = stack.pop()
            if (r, c) in component_visited:
                continue
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r][c] != color:
                continue

            component_visited.add((r, c))
            cells.append((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((r + dr, c + dc))

        return cells

    # Find all shapes
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                cells = flood_fill(r, c, color)
                visited.update(cells)

                if cells:
                    min_r = min(cell[0] for cell in cells)
                    max_r = max(cell[0] for cell in cells)
                    min_c = min(cell[1] for cell in cells)
                    max_c = max(cell[1] for cell in cells)

                    shapes.append({
                        "cells": cells,
                        "color": color,
                        "min_r": min_r,
                        "max_r": max_r,
                        "min_c": min_c,
                        "max_c": max_c,
                        "height": max_r - min_r + 1,
                        "center_r": (min_r + max_r) / 2,
                        "center_c": (min_c + max_c) / 2,
                    })

    # Group shapes by color
    from collections import defaultdict
    color_groups = defaultdict(list)
    for shape in shapes:
        color_groups[shape["color"]].append(shape)

    # Determine grid vertical center
    grid_center = rows / 2

    # Assign shapes to top or bottom output
    top_shapes = []
    bottom_shapes = []

    # Determine horizontal center
    grid_h_center = cols / 2

    # Count paired vs single colors
    paired_colors = [c for c, g in color_groups.items() if len(g) == 2]
    single_colors = [c for c, g in color_groups.items() if len(g) == 1]

    # Rule: if more paired colors than single, single components don't swap
    single_components_stay_in_place = len(paired_colors) > len(single_colors)

    for color, group in color_groups.items():
        if len(group) == 2:
            # Two components of same color
            # Sort by (min_r, min_c) to determine "first" and "second"
            group.sort(key=lambda s: (s["min_r"], s["min_c"]))
            shape1, shape2 = group

            # Check if both are in same input vertical half
            in_top_half_1 = shape1["center_r"] < grid_center
            in_top_half_2 = shape2["center_r"] < grid_center

            # Check if both are in same horizontal half
            in_left_half_1 = shape1["center_c"] < grid_h_center
            in_left_half_2 = shape2["center_c"] < grid_h_center

            same_v_half = (in_top_half_1 == in_top_half_2)
            same_h_half = (in_left_half_1 == in_left_half_2)

            if same_v_half:
                # Both in same vertical half - split between output top/bottom
                if color % 2 == 1:  # Odd color
                    top_shapes.append(shape1)
                    bottom_shapes.append(shape2)
                else:  # Even color
                    bottom_shapes.append(shape1)
                    top_shapes.append(shape2)
            elif not same_h_half:
                # Different vertical halves AND different horizontal sides
                if color % 2 == 1:  # Odd color - stay in place
                    if in_top_half_1:
                        top_shapes.append(shape1)
                    else:
                        bottom_shapes.append(shape1)
                    if in_top_half_2:
                        top_shapes.append(shape2)
                    else:
                        bottom_shapes.append(shape2)
                else:  # Even color - swap
                    if in_top_half_1:
                        bottom_shapes.append(shape1)
                    else:
                        top_shapes.append(shape1)
                    if in_top_half_2:
                        bottom_shapes.append(shape2)
                    else:
                        top_shapes.append(shape2)
            else:
                # Different vertical halves AND same horizontal side
                if color % 2 == 1:  # Odd color - both to bottom
                    bottom_shapes.append(shape1)
                    bottom_shapes.append(shape2)
                else:  # Even color - swap
                    if in_top_half_1:
                        bottom_shapes.append(shape1)
                    else:
                        top_shapes.append(shape1)
                    if in_top_half_2:
                        bottom_shapes.append(shape2)
                    else:
                        top_shapes.append(shape2)
        else:
            # Single component or more than 2
            for shape in group:
                if single_components_stay_in_place:
                    # Stay in place
                    if shape["center_r"] < grid_center:
                        top_shapes.append(shape)
                    else:
                        bottom_shapes.append(shape)
                else:
                    # Swap
                    if shape["center_r"] < grid_center:
                        bottom_shapes.append(shape)
                    else:
                        top_shapes.append(shape)

    # Sort shapes by color for consistent ordering
    top_shapes.sort(key=lambda s: (s["color"], s["min_r"], s["min_c"]))
    # For bottom shapes, sort by original row position (descending) to place lower shapes first
    bottom_shapes.sort(key=lambda s: (-s["min_r"], s["min_c"]))

    def pack_and_place_shapes(shapes_list, start_row, place_from_top=True):
        """Pack shapes efficiently and place in result grid"""
        if not shapes_list:
            return

        # Track which cells are occupied for each row
        occupied = {}  # row -> list of (min_c, max_c) ranges

        # Place each shape one by one
        for shape in shapes_list:
            # Find the best position for this shape
            if place_from_top:
                # Try to place as close to start_row as possible
                best_row = start_row
                # Check if we need to move down to avoid conflicts
                for test_row in range(start_row, rows):
                    # Check if shape fits at test_row
                    conflicts = False
                    for offset in range(shape["height"]):
                        check_row = test_row + offset
                        if check_row >= rows:
                            conflicts = True
                            break
                        if check_row in occupied:
                            for (occ_min, occ_max) in occupied[check_row]:
                                if not (shape["max_c"] < occ_min or shape["min_c"] > occ_max):
                                    conflicts = True
                                    break
                        if conflicts:
                            break
                    if not conflicts:
                        best_row = test_row
                        break
            else:
                # Try to place as close to start_row as possible (from bottom)
                best_row = start_row
                for test_row in range(start_row, -1, -1):
                    # Check if shape fits at test_row (as the bottom of the shape)
                    conflicts = False
                    for offset in range(shape["height"]):
                        check_row = test_row - offset
                        if check_row < 0:
                            conflicts = True
                            break
                        if check_row in occupied:
                            for (occ_min, occ_max) in occupied[check_row]:
                                if not (shape["max_c"] < occ_min or shape["min_c"] > occ_max):
                                    conflicts = True
                                    break
                        if conflicts:
                            break
                    if not conflicts:
                        best_row = test_row
                        break

            # Place the shape at best_row
            for r, c in shape["cells"]:
                if place_from_top:
                    offset = r - shape["min_r"]
                    new_r = best_row + offset
                else:
                    offset = shape["max_r"] - r
                    new_r = best_row - offset

                if 0 <= new_r < rows and 0 <= c < cols:
                    result[new_r][c] = shape["color"]
                    # Mark this cell as occupied
                    if new_r not in occupied:
                        occupied[new_r] = []
                    # Add to occupied ranges (could optimize by merging)
                    occupied[new_r].append((c, c))

    # Place top shapes starting from row 0
    pack_and_place_shapes(top_shapes, 0, place_from_top=True)

    # Place bottom shapes starting from last row
    pack_and_place_shapes(bottom_shapes, rows - 1, place_from_top=False)

    return result
