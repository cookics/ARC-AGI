def solve(grid: list[list[int]]) -> list[list[int]]:
    """
    Observation:
    1. Input: 20x20 grid, divide into 4 quadrants at row/col 10
    2. Each quadrant has a colored region (extracted as bounding box)
    3. One region has marker 5 at specific position
    4. Output: 4 regions tiled in 2x2 with marker at (0,0)
    5. Quadrants remap so marker region becomes top-left

    Procedure:
    1. Find and extract all 4 colored region bounding boxes
    2. Determine each region's input quadrant
    3. Remap quadrants so marker region is top-left
    4. Compute output size from tiled regions
    5. Place regions with marker at (0,0) and others tiled accordingly
    """

    from collections import defaultdict

    rows, cols = len(grid), len(grid[0])

    # Find marker position
    marker_r, marker_c = None, None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 5:
                marker_r, marker_c = r, c
                break
        if marker_r is not None:
            break

    if marker_r is None:
        return [[]]

    # Collect cells by color
    color_cells = defaultdict(list)
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != 0:
                color_cells[grid[r][c]].append((r, c))

    # Merge marker 5 with its neighbor color
    if 5 in color_cells:
        neighbor_color = None
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = marker_r + dr, marker_c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] != 0 and grid[nr][nc] != 5:
                    neighbor_color = grid[nr][nc]
                    break
        if neighbor_color:
            color_cells[neighbor_color].extend(color_cells[5])
            del color_cells[5]

    # Extract regions with bbox
    regions = {}
    for color, cells in color_cells.items():
        min_r, max_r = min(r for r,c in cells), max(r for r,c in cells)
        min_c, max_c = min(c for r,c in cells), max(c for r,c in cells)

        # Determine quadrant
        center_r, center_c = (min_r + max_r) / 2, (min_c + max_c) / 2
        quad = ("T" if center_r < 10 else "B") + ("L" if center_c < 10 else "R")

        # Extract bbox
        bbox = [[grid[r][c] for c in range(min_c, max_c + 1)] for r in range(min_r, max_r + 1)]

        # Check if has marker
        has_marker = any(grid[r][c] == 5 for r, c in cells)
        marker_in_bbox = (marker_r - min_r, marker_c - min_c) if has_marker else None

        regions[quad] = {
            "bbox": bbox,
            "has_marker": has_marker,
            "marker_pos": marker_in_bbox,
            "color": color
        }

    # Find marker quadrant
    marker_quad = None
    for quad, reg in regions.items():
        if reg["has_marker"]:
            marker_quad = quad
            break

    if not marker_quad:
        return [[]]

    # Remap quadrants (rotate so marker is TL)
    remap = {
        "TL": {"TL": "TL", "TR": "TR", "BL": "BL", "BR": "BR"},
        "TR": {"TL": "TR", "TR": "BR", "BL": "TL", "BR": "BL"},
        "BL": {"TL": "BL", "TR": "TL", "BL": "BR", "BR": "TR"},
        "BR": {"TL": "BR", "TR": "BL", "BL": "TR", "BR": "TL"}
    }[marker_quad]

    # Remap regions to output quadrants
    output_regions = {}
    for in_quad, reg in regions.items():
        out_quad = remap[in_quad]
        output_regions[out_quad] = reg

    # Get bboxes for each output quadrant
    tl = output_regions.get("TL", {"bbox": [[]]})["bbox"]
    tr = output_regions.get("TR", {"bbox": [[]]})["bbox"]
    bl = output_regions.get("BL", {"bbox": [[]]})["bbox"]
    br = output_regions.get("BR", {"bbox": [[]]})["bbox"]

    tl_h, tl_w = len(tl), len(tl[0]) if tl and tl[0] else 0
    tr_h, tr_w = len(tr), len(tr[0]) if tr and tr[0] else 0
    bl_h, bl_w = len(bl), len(bl[0]) if bl and bl[0] else 0
    br_h, br_w = len(br), len(br[0]) if br and br[0] else 0

    # Calculate output size
    out_h = max(tl_h, tr_h) + max(bl_h, br_h)
    out_w = max(tl_w, bl_w) + max(tr_w, br_w)

    # Create output
    result = [[0] * out_w for _ in range(out_h)]

    # Place TL region with marker at (0,0)
    if output_regions.get("TL", {}).get("has_marker"):
        mr, mc = output_regions["TL"]["marker_pos"]
        start_r, start_c = -mr, -mc
    else:
        start_r, start_c = 0, 0

    # Helper to place bbox
    def place(bbox, start_r, start_c):
        for r in range(len(bbox)):
            for c in range(len(bbox[0]) if bbox else 0):
                out_r, out_c = start_r + r, start_c + c
                if 0 <= out_r < out_h and 0 <= out_c < out_w:
                    if bbox[r][c] != 0:
                        result[out_r][out_c] = bbox[r][c]

    # Place all regions
    tl_start_r = -output_regions.get("TL", {}).get("marker_pos", (0, 0))[0] if output_regions.get("TL", {}).get("has_marker") else 0
    tl_start_c = -output_regions.get("TL", {}).get("marker_pos", (0, 0))[1] if output_regions.get("TL", {}).get("has_marker") else 0

    tr_start_r, tr_start_c = 0, max(tl_w, bl_w)
    bl_start_r, bl_start_c = max(tl_h, tr_h), 0
    br_start_r, br_start_c = max(tl_h, tr_h), max(tl_w, bl_w)

    place(bl, bl_start_r, bl_start_c)
    place(br, br_start_r, br_start_c)
    place(tr, tr_start_r, tr_start_c)
    place(tl, tl_start_r, tl_start_c)

    # Crop to non-zero bounding box
    non_zero = [(r, c) for r in range(out_h) for c in range(out_w) if result[r][c] != 0]
    if not non_zero:
        return [[]]

    min_r = min(r for r, c in non_zero)
    max_r = max(r for r, c in non_zero)
    min_c = min(c for r, c in non_zero)
    max_c = max(c for r, c in non_zero)

    return [[result[r][c] for c in range(min_c, max_c + 1)] for r in range(min_r, max_r + 1)]
