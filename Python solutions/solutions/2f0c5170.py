def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input contains two rectangular regions filled with 0s (background is 8s)
    2. One region has a single marker value (like 2, 3, or 1)
    3. The other region has a complex pattern including the same marker
    4. Output size is based on the marker-only region dimensions
    5. Pattern is overlaid with markers aligned

    Procedure:
    1. Find all rectangular 0-regions
    2. Identify marker region (single non-zero) vs pattern region (complex pattern)
    3. Calculate output size from marker region (adjust if width-height diff >= 2)
    4. Overlay pattern onto output, aligning marker values
    """

    # Find all 0-regions
    height, width = len(grid), len(grid[0])
    visited = [[False] * width for _ in range(height)]
    regions = []

    def find_region(start_r, start_c):
        """Find rectangular region containing non-8 values"""
        if visited[start_r][start_c] or grid[start_r][start_c] == 8:
            return None

        # Find connected component of non-8 values using BFS
        from collections import deque

        queue = deque([(start_r, start_c)])
        component = []

        while queue:
            r, c = queue.popleft()
            if r < 0 or r >= height or c < 0 or c >= width:
                continue
            if visited[r][c] or grid[r][c] == 8:
                continue

            visited[r][c] = True
            component.append((r, c))

            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                queue.append((r + dr, c + dc))

        if not component:
            return None

        # Find bounding box
        min_r = min(r for r, c in component)
        max_r = max(r for r, c in component)
        min_c = min(c for r, c in component)
        max_c = max(c for r, c in component)

        # Extract region content
        region_height = max_r - min_r + 1
        region_width = max_c - min_c + 1
        content = []

        for r in range(min_r, max_r + 1):
            row = []
            for c in range(min_c, max_c + 1):
                row.append(grid[r][c])
            content.append(row)

        return {
            "bounds": (min_r, max_r, min_c, max_c),
            "size": (region_height, region_width),
            "content": content,
            "component_size": len(component),
        }

    # Find all non-8 regions
    for r in range(height):
        for c in range(width):
            if grid[r][c] != 8 and not visited[r][c]:
                region = find_region(r, c)
                if region:
                    regions.append(region)

    # Filter to get the two largest regions (by component size)
    regions.sort(key=lambda x: x["component_size"], reverse=True)
    regions = regions[:2]

    assert len(regions) == 2, (
        f"Expected 2 regions after filtering, found {len(regions)}"
    )

    # Analyze regions to identify marker vs pattern
    def analyze_region(region):
        """Count non-zero values and find their positions"""
        non_zeros = []
        for r, row in enumerate(region["content"]):
            for c, val in enumerate(row):
                if val != 0:
                    non_zeros.append((val, r, c))
        return non_zeros

    region1_nonzeros = analyze_region(regions[0])
    region2_nonzeros = analyze_region(regions[1])

    # Determine which is marker region (single non-zero) vs pattern region
    if len(region1_nonzeros) == 1:
        marker_region = regions[0]
        pattern_region = regions[1]
        marker_value, marker_r, marker_c = region1_nonzeros[0]
    elif len(region2_nonzeros) == 1:
        marker_region = regions[1]
        pattern_region = regions[0]
        marker_value, marker_r, marker_c = region2_nonzeros[0]
    else:
        # If both regions have multiple non-zeros, find which has fewer
        if len(region1_nonzeros) < len(region2_nonzeros):
            marker_region = regions[0]
            pattern_region = regions[1]
            # Find the marker value (should be the one that appears in both)
            pattern_values = {val for val, _, _ in region2_nonzeros}
            for val, r, c in region1_nonzeros:
                if val in pattern_values:
                    marker_value, marker_r, marker_c = val, r, c
                    break
        else:
            marker_region = regions[1]
            pattern_region = regions[0]
            # Find the marker value (should be the one that appears in both)
            pattern_values = {val for val, _, _ in region1_nonzeros}
            for val, r, c in region2_nonzeros:
                if val in pattern_values:
                    marker_value, marker_r, marker_c = val, r, c
                    break

    # Calculate output size from marker region
    marker_height, marker_width = marker_region["size"]
    output_height = marker_height
    output_width = marker_width

    # Adjust if width-height difference >= 2 (make more square)
    if abs(marker_width - marker_height) >= 2:
        if marker_width > marker_height:
            output_width = marker_height
        else:
            output_height = marker_width

    # Scale marker position if output size changed
    scaled_marker_r = int(marker_r * output_height / marker_height)
    scaled_marker_c = int(marker_c * output_width / marker_width)

    # Find marker position in pattern region
    pattern_marker_r = pattern_marker_c = None
    for r, row in enumerate(pattern_region["content"]):
        for c, val in enumerate(row):
            if val == marker_value:
                pattern_marker_r, pattern_marker_c = r, c
                break
        if pattern_marker_r is not None:
            break

    assert pattern_marker_r is not None, "Marker value not found in pattern region"

    # Calculate shift to align markers
    shift_r = scaled_marker_r - pattern_marker_r
    shift_c = scaled_marker_c - pattern_marker_c

    # Create output grid
    result = [[0] * output_width for _ in range(output_height)]

    # Apply pattern with shift
    pattern_height, pattern_width = pattern_region["size"]
    for r in range(pattern_height):
        for c in range(pattern_width):
            new_r = r + shift_r
            new_c = c + shift_c
            if 0 <= new_r < output_height and 0 <= new_c < output_width:
                result[new_r][new_c] = pattern_region["content"][r][c]

    return result
