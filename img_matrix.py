import os
import json
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FOLDERS = {
    "V1": {
        "preds": os.path.join(BASE_DIR, "arc_agi_v1_public_eval"),
        "truth": os.path.join(BASE_DIR, "ARC-AGI", "data", "evaluation"),
        "title": "ARC-AGI-1 Public Eval Matrix (400 Tasks)"
    },
    "V2": {
        "preds": os.path.join(BASE_DIR, "arc_agi_v2_public_eval"),
        "truth": os.path.join(BASE_DIR, "ARC-AGI-2", "data", "evaluation"),
        "title": "ARC-AGI-2 Public Eval Matrix (120 Tasks)"
    }
}

# --- 2. SCORING HELPERS ---
def normalize_grid(grid):
    if not isinstance(grid, list): return []
    try:
        return [[int(cell) for cell in row] for row in grid]
    except:
        return []

def check_answer(pred_grid, true_grid):
    norm_pred = normalize_grid(pred_grid)
    norm_true = normalize_grid(true_grid)
    if not norm_pred or not norm_true: return False
    return norm_pred == norm_true

def build_results_matrix(preds_dir, truth_dir):
    """Returns a DataFrame where Index=Model, Columns=TaskID, Value=1 (Solved) or 0 (Failed)."""
    
    if not os.path.exists(preds_dir) or not os.path.exists(truth_dir):
        print(f"❌ Missing folder: {preds_dir} or {truth_dir}")
        return None

    # Get all Ground Truth Task IDs first to ensure consistent columns
    truth_files = glob.glob(os.path.join(truth_dir, "*.json"))
    all_task_ids = sorted([os.path.basename(f) for f in truth_files])
    
    if not all_task_ids:
        print("❌ No ground truth files found.")
        return None

    model_folders = [f for f in os.listdir(preds_dir) if os.path.isdir(os.path.join(preds_dir, f))]
    data = {}

    print(f"Processing {len(model_folders)} models against {len(all_task_ids)} tasks...")

    for model_name in model_folders:
        print(f"   Scoring {model_name}...", end="\r")
        model_path = os.path.join(preds_dir, model_name)
        
        # Initialize this model's row with 0s
        model_results = {task_id: 0 for task_id in all_task_ids}
        
        # Load all prediction files for this model
        pred_files = glob.glob(os.path.join(model_path, "*.json"))
        
        for pred_file in pred_files:
            task_id = os.path.basename(pred_file)
            
            # Skip if this task isn't in our ground truth list (e.g. extra files)
            if task_id not in model_results:
                continue

            # Load Pred
            try:
                with open(pred_file, 'r') as f:
                    pred_data = json.load(f)
            except:
                continue

            # Load Truth
            truth_path = os.path.join(truth_dir, task_id)
            try:
                with open(truth_path, 'r') as f:
                    truth_data = json.load(f)
            except:
                continue

            # Check Logic
            is_task_solved = True
            test_pairs = truth_data.get("test", [])
            
            for i, pair in enumerate(test_pairs):
                true_output = pair['output']
                
                # Fallback Strategy (Gemini Fix)
                pred_entry = next((item for item in pred_data if item.get("metadata", {}).get("pair_index") == i), None)
                if not pred_entry and i < len(pred_data):
                    pred_entry = pred_data[i]

                if not pred_entry:
                    is_task_solved = False
                    break
                
                # Check Attempt 1 then 2
                attempt1 = pred_entry.get("attempt_1") or {}
                model_output = attempt1.get("answer")
                
                if not model_output:
                     attempt2 = pred_entry.get("attempt_2") or {}
                     model_output = attempt2.get("answer")

                if not check_answer(model_output, true_output):
                    is_task_solved = False
                    break
            
            if is_task_solved:
                model_results[task_id] = 1
        
        data[model_name] = model_results

    print(" " * 50, end="\r") # Clear line
    return pd.DataFrame.from_dict(data, orient='index')

# --- 3. VISUALIZATION ---
def plot_matrix(df, title, filename):
    if df is None or df.empty:
        print(f"Skipping plot for {title} (No data)")
        return

    # SORTING for better visualization:
    # 1. Sort Tasks (Columns) by difficulty (Sum of solved count), Descending
    df = df[df.sum().sort_values(ascending=False).index]
    
    # 2. Sort Models (Rows) by accuracy (Sum of solved count), Descending
    df = df.loc[df.sum(axis=1).sort_values(ascending=False).index]

    # Create Plot
    plt.figure(figsize=(24, 12)) # Large canvas
    
    # Custom Red/Green Colormap (0=Red, 1=Green)
    cmap = ListedColormap(['#ff9999', '#99ff99']) # Pastel Red / Pastel Green
    
    ax = sns.heatmap(df, cmap=cmap, cbar=False, linewidths=0.0, linecolor='white')
    
    # Styling
    plt.title(title, fontsize=20, pad=20)
    plt.xlabel(f"Tasks (Sorted: Easy -> Hard)", fontsize=14)
    plt.ylabel("Models (Sorted: Best -> Worst)", fontsize=14)
    
    # Remove x-axis tick labels (too many tasks to read them all)
    ax.set_xticks([]) 
    
    # Ensure Y-axis labels (Model names) are readable
    plt.yticks(rotation=0, fontsize=10)
    
    # Save
    save_path = os.path.join(BASE_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved Image: {save_path}")
    plt.close()

# --- MAIN ---
if __name__ == "__main__":
    print("Generating Matrices...")
    
    # Process V1
    df_v1 = build_results_matrix(FOLDERS["V1"]["preds"], FOLDERS["V1"]["truth"])
    plot_matrix(df_v1, FOLDERS["V1"]["title"], "arc_v1_matrix.png")
    
    # Process V2
    df_v2 = build_results_matrix(FOLDERS["V2"]["preds"], FOLDERS["V2"]["truth"])
    plot_matrix(df_v2, FOLDERS["V2"]["title"], "arc_v2_matrix.png")
    
    print("\nDone! Check your folder for the PNG files.")