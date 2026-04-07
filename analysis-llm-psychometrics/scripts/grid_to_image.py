import json
import os
from PIL import Image, ImageDraw

# Standard ARC Color Palette
ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: gray
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: light blue
    "#870C25",  # 9: maroon
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data-llm")

def render_grid(grid, cell_size=30, border_width=1):
    rows = len(grid)
    cols = len(grid[0])
    
    width = cols * cell_size
    height = rows * cell_size
    
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            color = ARC_COLORS[val] if val < len(ARC_COLORS) else "#FFFFFF"
            
            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            
            # Draw cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="#333333", width=border_width)
            
    return img

def main():
    task_id = "62593bfd"
    input_file = os.path.join(DATA_DIR, "ARC-AGI-2", "data", "evaluation", f"{task_id}.json")
    output_dir = os.path.join(PROJECT_DIR, "output_images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Process Train Pairs
    for i, pair in enumerate(data.get("train", [])):
        for key in ["input", "output"]:
            grid = pair.get(key)
            if grid:
                img = render_grid(grid)
                filename = f"{task_id}_train_{i}_{key}.png"
                img.save(os.path.join(output_dir, filename))
                print(f"Saved {filename}")
                
    # Process Test Pairs
    for i, pair in enumerate(data.get("test", [])):
        for key in ["input", "output"]:
            grid = pair.get(key)
            if grid:
                img = render_grid(grid)
                filename = f"{task_id}_test_{i}_{key}.png"
                img.save(os.path.join(output_dir, filename))
                print(f"Saved {filename}")

if __name__ == "__main__":
    main()
