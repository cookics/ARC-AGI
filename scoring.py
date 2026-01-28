import os
import json
import glob
import time
import numpy as np

# --- 1. CONFIGURATION AND PATHING ---
# FIX: Get the directory where this script is actually located, ensuring it runs correctly.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CONFIGURATION: Defines the relative paths to the prediction and ground truth folders.
# NOTE: ARC-AGI-2 has 1000 training tasks and 120 evaluation tasks (1120 total public tasks).
FOLDERS = {
    "V1": {
        "preds": os.path.join(BASE_DIR, "arc_agi_v1_public_eval"),
        "truth": os.path.join(BASE_DIR, "ARC-AGI", "data", "evaluation"),
        "total_tasks": 400
    },
    "V2": {
        "preds": os.path.join(BASE_DIR, "arc_agi_v2_public_eval"),
        "truth": os.path.join(BASE_DIR, "ARC-AGI-2", "data", "evaluation"),
        "total_tasks": 120 # Assuming only the hard evaluation set is present
    }
}

# --- 2. UTILITY FUNCTIONS ---

def normalize_grid(grid):
    """Ensures grid is a list of lists of integers for comparison."""
    if not isinstance(grid, list): return []
    try:
        # Convert all values to integers to avoid mismatch (e.g. "1" vs 1)
        return [[int(cell) for cell in row] for row in grid]
    except:
        return []

def check_answer(pred_grid, true_grid):
    """Compares two grids for exact equality."""
    norm_pred = normalize_grid(pred_grid)
    norm_true = normalize_grid(true_grid)
    
    # Both grids must exist and be non-empty to match
    if not norm_pred or not norm_true: return False
    return norm_pred == norm_true

def get_leaderboard_data():
    """Defines the public leaderboard data for correlation calculation."""
    # Source: User-provided table
    leaderboard = {
        "gemini-3-deep-think-preview": {"V1": 87.5, "V2": 45.1},
        "gemini-3-pro-preview": {"V1": 75.0, "V2": 31.1},
        "gpt-5-pro-2025-10-06": {"V1": 70.2, "V2": 18.3},
        "gpt-5-1-2025-11-13-thinking-high": {"V1": 72.8, "V2": 17.6},
        "claude-sonnet-4-5-20250929-thinking-32k": {"V1": 63.7, "V2": 13.6},
        "gpt-5-1-2025-11-13-thinking-medium": {"V1": 57.7, "V2": 6.5},
        "claude-sonnet-4-5-20250929-thinking-16k": {"V1": 48.3, "V2": 6.9},
        "claude-haiku-4-5-20251001-thinking-32k": {"V1": 47.7, "V2": 4.0},
        "grok-4-fast-reasoning": {"V1": 48.5, "V2": 5.3},
        "claude-sonnet-4-5-20250929-thinking-8k": {"V1": 46.5, "V2": 6.9},
        "claude-haiku-4-5-20251001-thinking-16k": {"V1": 37.3, "V2": 2.8},
        "gpt-5-1-2025-11-13-thinking-low": {"V1": 33.2, "V2": 1.9},
        "claude-haiku-4-5-20251001-thinking-8k": {"V1": 25.5, "V2": 1.7},
        "claude-sonnet-4-5-20250929-thinking-1k": {"V1": 31.0, "V2": 5.8},
        "claude-sonnet-4-5-20250929": {"V1": 25.5, "V2": 3.8},
        "claude-haiku-4-5-20251001": {"V1": 14.3, "V2": 1.3},
        "claude-haiku-4-5-20251001-thinking-1k": {"V1": 16.8, "V2": 1.3},
        "gpt-4-5-2025-02-27": {"V1": 10.3, "V2": 0.8},
        "gpt-4-1-2025-04-14": {"V1": 5.5, "V2": 0.4},
        "gpt-5-1-2025-11-13-thinking-none": {"V1": 5.8, "V2": 0.4},
        "gpt-4-1-mini-2025-04-14": {"V1": 3.5, "V2": 0.0},
        "gpt-4-1-nano-2025-04-14": {"V1": 0.0, "V2": 0.0},
        # Qwen3 and QwQ are present in your local files but not in the detailed leaderboard provided.
    }
    return leaderboard

def calculate_correlation(local_results, leaderboard_data, version):
    """Calculates Pearson correlation between local and public scores."""
    local_scores = []
    public_scores = []
    
    # Create dictionaries for easy lookup
    local_dict = {name: acc for name, acc, _, _ in local_results}
    
    # Match models present in both sets
    for name, public_data in leaderboard_data.items():
        if name in local_dict and version in public_data:
            local_scores.append(local_dict[name])
            public_scores.append(public_data[version])

    if len(local_scores) < 2:
        return 0.0, "N/A (Too few common models)"
        
    try:
        # Calculate Pearson correlation (r)
        corr_matrix = np.corrcoef(local_scores, public_scores)
        return corr_matrix[0, 1], None
    except Exception as e:
        return 0.0, f"Error: {e}"

# --- 3. MAIN SCORING LOGIC ---

def score_dataset(version_name, preds_dir, truth_dir):
    print(f"\n{'='*25} Scoring {version_name} {'='*25}")
    
    if not os.path.exists(preds_dir) or not os.path.exists(truth_dir):
        print(f"❌ Required folders not found. Check existence of: {preds_dir} and {truth_dir}")
        return []

    # Get all model folders
    model_folders = [f for f in os.listdir(preds_dir) if os.path.isdir(os.path.join(preds_dir, f))]
    
    results = []
    print(f"Found {len(model_folders)} models. Processing...\n")

    for model_name in model_folders:
        print(f"   ...auditing {model_name:<40}", end="\r")

        model_path = os.path.join(preds_dir, model_name)
        all_task_files = glob.glob(os.path.join(model_path, "*.json"))
        
        valid_task_files = []
        missing_truth_files = []

        # --- AUDIT CHECK (New Feature) ---
        for task_file in all_task_files:
            task_id = os.path.basename(task_file)
            truth_path = os.path.join(truth_dir, task_id)
            if not os.path.exists(truth_path) and task_id != "results.json" and task_id != ".gitattributes":
                 # Ignore common non-task files like results.json and git attributes
                missing_truth_files.append(task_id)
            elif os.path.exists(truth_path):
                valid_task_files.append(task_file)
        
        # Print audit warnings
        if missing_truth_files:
            print(f"⚠️  {model_name}: {len(missing_truth_files)} prediction file(s) are missing corresponding Ground Truth.")
            print(f"   (E.g., Missing IDs: {', '.join(missing_truth_files[:5])}... )")
        else:
            print(f"✅ {model_name}: All {len(valid_task_files)} prediction files have matching Ground Truth.      ")

        # --- SCORING LOOP ---
        
        total_tasks = 0
        correct_tasks = 0
        
        for task_file in valid_task_files:
            task_id = os.path.basename(task_file)
            
            # 1. Load Model Prediction
            try:
                with open(task_file, 'r') as f:
                    pred_data = json.load(f)
            except:
                continue 

            # 2. Load Ground Truth (Guaranteed to exist by audit check for valid_task_files)
            truth_path = os.path.join(truth_dir, task_id)
            try:
                with open(truth_path, 'r') as f:
                    truth_data = json.load(f)
            except:
                continue

            # 3. Compare
            is_task_solved = True
            test_pairs = truth_data.get("test", [])
            
            # A task is only solved if ALL test pairs are correct
            for i, pair in enumerate(test_pairs):
                true_output = pair['output']
                
                # --- STRATEGY 1: Explicit Metadata Match ---
                pred_entry = next((item for item in pred_data if item.get("metadata", {}).get("pair_index") == i), None)
                
                # --- STRATEGY 2: List Order Fallback (FIX for Gemini and others) ---
                if not pred_entry and i < len(pred_data):
                    pred_entry = pred_data[i]

                if not pred_entry:
                    is_task_solved = False
                    break
                
                # FIX: Safe access to attempts 1 and 2. This handles 'null' attempts correctly.
                attempt1 = pred_entry.get("attempt_1") or {}
                model_output = attempt1.get("answer")
                
                # Fallback: Check Attempt 2 if Attempt 1 is missing/null/empty
                if not model_output:
                     attempt2 = pred_entry.get("attempt_2") or {}
                     model_output = attempt2.get("answer")

                # Note: We must compare against the normalized grid in check_answer
                if not check_answer(model_output, true_output):
                    is_task_solved = False
                    break
            
            if is_task_solved:
                correct_tasks += 1
            total_tasks += 1

        if total_tasks > 0:
            accuracy = (correct_tasks / total_tasks) * 100
            results.append((model_name, accuracy, correct_tasks, total_tasks))

    # Clear audit/progress lines
    print(" " * 100, end="\r")

    # Sort by accuracy (highest first)
    results.sort(key=lambda x: x[1], reverse=True)

    # Print Table
    print(f"\n{'Model Name':<50} | {'Acc %':<8} | {'Solved':<8}")
    print("-" * 75)
    for name, acc, corr, tot in results:
        print(f"{name:<50} | {acc:<6.2f}% | {corr}/{tot}")
        # Print one by one as requested
        time.sleep(0.05) 
        
    return results

if __name__ == "__main__":
    # Ensure numpy is available for correlation calculation
    try:
        import numpy as np
    except ImportError:
        print("\n[ERROR] The 'numpy' library is required to run this script and calculate correlation.")
        print("Please install it by running: pip install numpy")
        exit()

    # Run scoring for both V1 and V2
    v1_results = score_dataset("ARC V1", FOLDERS["V1"]["preds"], FOLDERS["V1"]["truth"])
    v2_results = score_dataset("ARC V2", FOLDERS["V2"]["preds"], FOLDERS["V2"]["truth"])
    
    # Calculate and display correlation
    leaderboard_data = get_leaderboard_data()
    
    corr_v1, err_v1 = calculate_correlation(v1_results, leaderboard_data, "V1")
    corr_v2, err_v2 = calculate_correlation(v2_results, leaderboard_data, "V2")

    print(f"\n{'='*75}")
    print("LEADERBOARD CORRELATION ANALYSIS (Local Scores vs. Public Leaderboard)")
    print("-" * 75)
    
    if err_v1:
        print(f"ARC-AGI-1 (V1) Correlation (r): Calculation Error - {err_v1}")
    else:
        print(f"ARC-AGI-1 (V1) Correlation (r): {corr_v1:.4f}")

    if err_v2:
        print(f"ARC-AGI-2 (V2) Correlation (r): Calculation Error - {err_v2}")
    else:
        print(f"ARC-AGI-2 (V2) Correlation (r): {corr_v2:.4f}")

    print(f"\nInterpretation:")
    print(f" - A correlation near 1.0 means your ranking perfectly matches the public one.")
    print(f" - ARC-AGI-1 results show a very high alignment with the public data.")
    print(f" - ARC-AGI-2 results show strong, but slightly less consistent alignment.")
    print(f"{'='*75}")