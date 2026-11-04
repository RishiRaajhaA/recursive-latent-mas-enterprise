"""
Performance and Computational Analysis CSV & Comparative Plot Generator
Features:
  - Task-wise grouping (Task ID -> Recursive Depths r=0, 1, 2, 3)
  - Added 'difficulty' column (Easy, Medium, Hard)
  - Model-wise separation (Text ReAct vs Latent ReAct)
  - Summary row: FLOPs and Output Tokens are AGGREGATED (Total sum), while Latency, Accuracy, and Tool Accuracy are AVERAGED (Mean)
"""
import csv
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEXT_CSV_PATH = os.path.join(RESULTS_DIR, "performance_and_computational_analysis_text_react.csv")
LATENT_CSV_PATH = os.path.join(RESULTS_DIR, "performance_and_computational_analysis_latent_react.csv")
ACCURACY_PLOT_PATH = os.path.join(RESULTS_DIR, "accuracy_vs_recursive_depth.png")
FLOPS_PLOT_PATH = os.path.join(RESULTS_DIR, "flops_vs_recursive_depth.png")

# Tier mapping fallback
TASK_TIER_MAP = {
    "24": "Easy",
    "57": "Easy",
    "67": "Medium",
    "100": "Medium",
    "7": "Hard",
    "8": "Hard"
}


def load_all_results() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load all JSONL result checkpoints, separating into text and latent records with deduplication."""
    text_dict = {}
    latent_dict = {}
    
    jsonl_files = glob.glob(os.path.join(RESULTS_DIR, "results_*.jsonl"))
    
    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        arch = record.get("architecture", "text_react")
                        d = record.get("recursive_depth", 0)
                        tid = str(record.get("task_id", ""))
                        key = (tid, d)
                        
                        if arch == "text_react":
                            text_dict[key] = record
                        else:
                            latent_dict[key] = record
                    except Exception:
                        pass
                        
    text_records = list(text_dict.values())
    latent_records = list(latent_dict.values())
    return text_records, latent_records


def get_task_difficulty(record: Dict[str, Any]) -> str:
    """Retrieve formatted difficulty for task."""
    tid = str(record.get("task_id", ""))
    tier = record.get("complexity_tier", "")
    if tier:
        return str(tier).capitalize()
    return TASK_TIER_MAP.get(tid, "Medium")


def export_single_method_csv(records: List[Dict[str, Any]], output_path: str, method_name: str) -> str:
    """Export benchmark records with difficulty column, grouped task-wise and model-wise."""
    fieldnames = [
        "recursive depth(r=0/1/2/3)",
        "task id",
        "difficulty",
        "latency",
        "FLOPs",
        "Accuracy",
        "Output tokens",
        "tool_selection_accuracy"
    ]
    
    # Sort strictly by Task ID first, then by recursive depth (0, 1, 2, 3)
    sorted_records = sorted(records, key=lambda x: (
        int(x.get("task_id")) if str(x.get("task_id", "")).isdigit() else str(x.get("task_id", "")),
        x.get("recursive_depth", 0)
    ))
    
    if not sorted_records:
        print(f"[CSV Warning] No records to export for {method_name}")
        return output_path
        
    all_latencies = [r.get("latency", 0.0) for r in sorted_records]
    all_flops = [r.get("FLOPs", 0.0) for r in sorted_records]
    all_accuracies = [r.get("Accuracy", 0.0) for r in sorted_records]
    all_tokens = [r.get("Output_tokens", 0) for r in sorted_records]
    all_tool_accs = [r.get("tool_selection_accuracy", 100.0) for r in sorted_records]
    
    # Averages for latency, Accuracy, Tool Selection Accuracy
    avg_latency = round(float(np.mean(all_latencies)), 3)
    avg_accuracy = round(float(np.mean(all_accuracies)), 1)
    avg_tool_acc = round(float(np.mean(all_tool_accs)), 1)
    
    # Aggregated totals (Sum) for FLOPs and Output Tokens
    total_flops = round(float(sum(all_flops)), 4)
    total_tokens = sum(all_tokens)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write individual task rows grouped by task
        for r in sorted_records:
            writer.writerow({
                "recursive depth(r=0/1/2/3)": r.get("recursive_depth", 0),
                "task id": r.get("task_id", ""),
                "difficulty": get_task_difficulty(r),
                "latency": r.get("latency", 0.0),
                "FLOPs": r.get("FLOPs", 0.0),
                "Accuracy": r.get("Accuracy", 0.0),
                "Output tokens": r.get("Output_tokens", 0),
                "tool_selection_accuracy": r.get("tool_selection_accuracy", 100.0)
            })
            
        # Summary Row: FLOPs and Tokens Aggregated (Sum), all others Averaged (Mean)
        writer.writerow({
            "recursive depth(r=0/1/2/3)": "Summary",
            "task id": "ALL",
            "difficulty": "All Tiers",
            "latency": avg_latency,
            "FLOPs": total_flops,
            "Accuracy": avg_accuracy,
            "Output tokens": total_tokens,
            "tool_selection_accuracy": avg_tool_acc
        })
        
    print(f"[CSV Export] Successfully written {len(sorted_records)} tasks + 1 summary row to {output_path}")
    return output_path


def generate_comparative_plots(text_records: List[Dict[str, Any]], latent_records: List[Dict[str, Any]]):
    """Generate high-resolution comparative publication plots."""
    depths = [0, 1, 2, 3]
    
    data_acc = {"text_react": {d: [] for d in depths}, "latent_react": {d: [] for d in depths}}
    data_flops = {"text_react": {d: [] for d in depths}, "latent_react": {d: [] for d in depths}}
    
    for r in text_records:
        d = r.get("recursive_depth", 0)
        if d in data_acc["text_react"]:
            data_acc["text_react"][d].append(r.get("Accuracy", 0.0))
            data_flops["text_react"][d].append(r.get("FLOPs", 0.0))
            
    for r in latent_records:
        d = r.get("recursive_depth", 0)
        if d in data_acc["latent_react"]:
            data_acc["latent_react"][d].append(r.get("Accuracy", 0.0))
            data_flops["latent_react"][d].append(r.get("FLOPs", 0.0))
            
    mean_acc_text = [np.mean(data_acc["text_react"][d]) if data_acc["text_react"][d] else 0.0 for d in depths]
    mean_acc_latent = [np.mean(data_acc["latent_react"][d]) if data_acc["latent_react"][d] else 0.0 for d in depths]
    
    mean_flops_text = [np.mean(data_flops["text_react"][d]) if data_flops["text_react"][d] else 0.0 for d in depths]
    mean_flops_latent = [np.mean(data_flops["latent_react"][d]) if data_flops["latent_react"][d] else 0.0 for d in depths]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {
        "text": "#2563EB",   # Royal Blue
        "latent": "#059669"  # Emerald Green
    }

    # --------------------------------------------------------------------------
    # Plot 1: Accuracy vs Recursive Depth
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(depths, mean_acc_text, marker='o', markersize=8, linewidth=2.4, linestyle='--', color=colors["text"], label="Text ReAct (Verbal Communication)")
    ax.plot(depths, mean_acc_latent, marker='s', markersize=8, linewidth=2.6, linestyle='-', color=colors["latent"], label="Latent ReAct (RecursiveLink Communication)")
    
    for d, val in zip(depths, mean_acc_text):
        if val > 0:
            ax.annotate(f"{val:.1f}%", (d, val), textcoords="offset points", xytext=(-10, 10), fontsize=9, color=colors["text"], fontweight="bold")
    for d, val in zip(depths, mean_acc_latent):
        if val > 0:
            ax.annotate(f"{val:.1f}%", (d, val), textcoords="offset points", xytext=(-10, -15), fontsize=9, color=colors["latent"], fontweight="bold")

    ax.set_title("Task Accuracy vs. Recursive Depth ($r = 0, 1, 2, 3$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recursive Depth ($r$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy / Goal Satisfaction (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(depths)
    ax.set_xticklabels([f"r = {d}" for d in depths])
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH)
    plt.close()
    print(f"[Plot 1] Saved Accuracy plot to {ACCURACY_PLOT_PATH}")

    # --------------------------------------------------------------------------
    # Plot 2: FLOPs vs Recursive Depth
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    ax.plot(depths, mean_flops_text, marker='o', markersize=8, linewidth=2.4, linestyle='--', color=colors["text"], label="Text ReAct (Full Text Tokens)")
    ax.plot(depths, mean_flops_latent, marker='s', markersize=8, linewidth=2.6, linestyle='-', color=colors["latent"], label="Latent ReAct (0 Intermediate Text Tokens)")
    
    for d, val in zip(depths, mean_flops_text):
        if val > 0:
            ax.annotate(f"{val:.2f}T", (d, val), textcoords="offset points", xytext=(-10, 10), fontsize=9, color=colors["text"], fontweight="bold")
    for d, val in zip(depths, mean_flops_latent):
        if val > 0:
            ax.annotate(f"{val:.2f}T", (d, val), textcoords="offset points", xytext=(-10, -15), fontsize=9, color=colors["latent"], fontweight="bold")

    ax.set_title("Computational Cost (FLOPs) vs. Recursive Depth ($r = 0, 1, 2, 3$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recursive Depth ($r$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Estimated Compute (TFLOPs)", fontsize=11, fontweight="bold")
    ax.set_xticks(depths)
    ax.set_xticklabels([f"r = {d}" for d in depths])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(FLOPS_PLOT_PATH)
    plt.close()
    print(f"[Plot 2] Saved FLOPs plot to {FLOPS_PLOT_PATH}")


def main():
    print("=" * 80)
    print("  GENERATING CSVs (WITH DIFFICULTY COLUMN, FLOPS & TOKENS AGGREGATED)")
    print("=" * 80)
    text_records, latent_records = load_all_results()
    
    export_single_method_csv(text_records, TEXT_CSV_PATH, "Text ReAct")
    export_single_method_csv(latent_records, LATENT_CSV_PATH, "Latent ReAct")
    generate_comparative_plots(text_records, latent_records)
    print("=" * 80)


if __name__ == "__main__":
    main()
