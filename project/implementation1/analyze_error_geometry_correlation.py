"""
Stage 1 - Extension: Error Analysis & Latent Geometric Correlation Inference
Investigates:
  1. Fine-grained categorization of failure modes:
     - Type A: Tool Selection Error (incorrect API choice)
     - Type B: Argument / Schema Validation Error (malformed parameters)
     - Type C: Policy Constraint Violation / State Tracking Loss
     - Type D: Premature Termination (incomplete goal fulfillment)
  2. Statistical Correlation (Pearson r & Spearman rho) between Latent Geometry (Drift, Distance, Alignment)
     and specific error occurrences.
  3. Predictive Latent Probing (Logistic Probe & ROC-AUC):
     Infers/predicts error occurrence directly from the latent representation vector H_r in R^2560.
  4. Publication Figure: error_modes_vs_geometry.png (300 DPI).

Outputs:
  - results/error_geometry_correlation.csv
  - results/error_modes_geometric_profiles.csv
  - results/error_inference_probe_results.json
  - results/error_modes_vs_geometry.png
"""
import glob
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LATENTS_DIR = os.path.join(RESULTS_DIR, "latents")
MASTER_CACHE_FILE = os.path.join(LATENTS_DIR, "all_representations.pt")
METRICS_CSV_FILE = os.path.join(RESULTS_DIR, "geometric_analysis_metrics.csv")

CORR_CSV_FILE = os.path.join(RESULTS_DIR, "error_geometry_correlation.csv")
PROFILES_CSV_FILE = os.path.join(RESULTS_DIR, "error_modes_geometric_profiles.csv")
PROBE_JSON_FILE = os.path.join(RESULTS_DIR, "error_inference_probe_results.json")
PLOT_PATH = os.path.join(RESULTS_DIR, "error_modes_vs_geometry.png")


def classify_task_error_mode(record: dict) -> str:
    """Classify the primary failure mode of a failed task execution."""
    if record.get("is_success", False):
        return "Success (No Error)"

    trajectories = record.get("round_trajectories", [])
    has_arg_err = False
    has_tool_sel_err = False

    for r_data in trajectories:
        tc = r_data.get("agent_2_toolcaller", {})
        if not tc.get("tool_call_valid", True):
            has_arg_err = True
        tool_call = tc.get("parsed_tool_call", {})
        if not tool_call.get("name"):
            has_arg_err = True

    tool_sel_acc = record.get("tool_selection_accuracy", 100.0)
    if tool_sel_acc < 80.0:
        has_tool_sel_err = True

    goal_score = record.get("goal_score", 0.0)

    if has_arg_err:
        return "Argument / Validation Error"
    elif has_tool_sel_err:
        return "Tool Selection Error"
    elif goal_score < 0.3:
        return "Policy Constraint Violation"
    else:
        return "Premature Termination"


def run_error_geometry_analysis():
    print("=" * 80)
    print("STAGE 1 EXTENSION: ERROR ANALYSIS & LATENT GEOMETRIC CORRELATION INFERENCE")
    print("=" * 80)

    # 1. Load results and annotate error modes
    jsonl_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.jsonl")))
    run_records = {}
    for fpath in jsonl_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    arch = rec.get("architecture", "text_react")
                    tid = str(rec.get("task_id", ""))
                    depth = int(rec.get("recursive_depth", 0))
                    run_records[(arch, tid, depth)] = rec

    df_metrics = pd.read_csv(METRICS_CSV_FILE)
    df_metrics["task_id"] = df_metrics["task_id"].astype(str)

    error_modes = []
    has_error_list = []
    is_tool_err_list = []
    is_arg_err_list = []
    is_policy_err_list = []

    for _, row in df_metrics.iterrows():
        arch = row["architecture"]
        tid = row["task_id"]
        # Look up r=3 execution
        rec = run_records.get((arch, tid, 3))
        if rec is None:
            # Fallback to r0 or highest depth
            rec = run_records.get((arch, tid, 0), {})

        err_mode = classify_task_error_mode(rec)
        error_modes.append(err_mode)

        is_err = 0 if err_mode == "Success (No Error)" else 1
        has_error_list.append(is_err)
        is_tool_err_list.append(1 if err_mode == "Tool Selection Error" else 0)
        is_arg_err_list.append(1 if err_mode == "Argument / Validation Error" else 0)
        is_policy_err_list.append(1 if err_mode == "Policy Constraint Violation" else 0)

    df_metrics["error_mode"] = error_modes
    df_metrics["has_error"] = has_error_list
    df_metrics["err_tool_selection"] = is_tool_err_list
    df_metrics["err_argument"] = is_arg_err_list
    df_metrics["err_policy"] = is_policy_err_list

    # 2. Geometric Profiles by Error Mode
    print("\n--- Geometric Profiles Across Error Modes ---")
    profiles = df_metrics.groupby(["architecture", "error_mode"]).agg({
        "task_id": "count",
        "drift_D1": "mean",
        "drift_D2": "mean",
        "drift_D3": "mean",
        "drift_decay_slope": "mean",
        "inter_agent_distance": "mean"
    }).rename(columns={"task_id": "count"}).reset_index()

    profiles.to_csv(PROFILES_CSV_FILE, index=False)
    print(f"Saved error mode geometric profiles to: {PROFILES_CSV_FILE}")
    for _, row in profiles.iterrows():
        print(f"  [{row['architecture']}] {row['error_mode']} (n={row['count']}): D1={row['drift_D1']:.4f}, D3={row['drift_D3']:.4f}, Slope={row['drift_decay_slope']:.4f}")

    # 3. Statistical Correlation Analysis (Pearson r & Spearman rho)
    print("\n--- Statistical Correlation Matrix (Latent Geometry vs Errors) ---")
    corr_results = []
    geo_features = ["drift_D1", "drift_D2", "drift_D3", "drift_decay_slope", "inter_agent_distance"]
    err_targets = [("Overall Task Failure", "has_error"),
                   ("Tool Selection Error", "err_tool_selection"),
                   ("Argument Validation Error", "err_argument"),
                   ("Policy Violation", "err_policy")]

    for arch in ["latent_react", "text_react"]:
        sub_df = df_metrics[df_metrics["architecture"] == arch]
        for feat in geo_features:
            for target_name, target_col in err_targets:
                if sub_df[target_col].std() > 0:
                    r_val, p_val = stats.pearsonr(sub_df[feat], sub_df[target_col])
                    rho_val, rho_p = stats.spearmanr(sub_df[feat], sub_df[target_col])
                    corr_results.append({
                        "architecture": arch,
                        "geometric_feature": feat,
                        "error_target": target_name,
                        "pearson_r": round(float(r_val), 4),
                        "pearson_p_value": round(float(p_val), 5),
                        "spearman_rho": round(float(rho_val), 4),
                        "spearman_p_value": round(float(rho_p), 5),
                        "is_significant": bool(p_val < 0.05)
                    })

    df_corr = pd.DataFrame(corr_results)
    df_corr.to_csv(CORR_CSV_FILE, index=False)
    print(f"Saved correlation analysis to: {CORR_CSV_FILE}")

    # Display top correlations
    sig_corr = df_corr[df_corr["is_significant"]]
    print(f"Total statistically significant correlations found: {len(sig_corr)}")
    for _, sc in sig_corr.head(5).iterrows():
        print(f"  [{sc['architecture']}] {sc['geometric_feature']} <-> {sc['error_target']}: r={sc['pearson_r']:.3f} (p={sc['pearson_p_value']:.4f})")

    # 4. Predictive Latent Probing (Error Inference from H_r in R^2560)
    print("\n--- Predictive Latent Probing (Inferring Errors directly from Latent Vectors) ---")
    database = torch.load(MASTER_CACHE_FILE, map_location="cpu", weights_only=False)

    probe_metrics = {}
    roc_data = {}

    for arch in ["latent_react", "text_react"]:
        sub_df = df_metrics[df_metrics["architecture"] == arch].copy()
        X_list = []
        y_list = []

        for _, row in sub_df.iterrows():
            tid = str(row["task_id"])
            key = (arch, tid, 3)
            if key in database and 3 in database[key]["rounds"]:
                vec = database[key]["rounds"][3]["Hr"].numpy()
                X_list.append(vec)
                y_list.append(row["has_error"])

        X = np.array(X_list)
        y = np.array(y_list)

        # 5-Fold Stratified Cross-Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_probs = np.zeros(len(y))

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            y_probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

        auc = roc_auc_score(y, y_probs)
        y_pred = (y_probs >= 0.5).astype(int)
        acc = float(np.mean(y_pred == y))
        f1 = float(f1_score(y, y_pred))

        fpr, tpr, _ = roc_curve(y, y_probs)
        roc_data[arch] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}

        probe_metrics[arch] = {
            "ROC_AUC": round(float(auc), 4),
            "Accuracy": round(float(acc) * 100.0, 2),
            "F1_Score": round(float(f1), 4),
            "sample_size": len(y),
            "error_rate_pct": round(float(np.mean(y)) * 100.0, 1)
        }
        print(f"  [{arch.upper()}] Error Inference Probe: ROC-AUC = {auc:.4f} | Accuracy = {acc*100.1:.1f}% | F1 = {f1:.4f}")

    with open(PROBE_JSON_FILE, "w", encoding="utf-8") as f_json:
        json.dump(probe_metrics, f_json, indent=2)
    print(f"Saved probe evaluation to: {PROBE_JSON_FILE}")

    # 5. Generate Publication Figure (error_modes_vs_geometry.png)
    print("\nGenerating Error vs Geometry Visualization Suite ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Subplot A: Drift Trajectories by Error Mode (Latent ReAct)
    ax1 = axes[0]
    lat_profiles = profiles[profiles["architecture"] == "latent_react"]
    colors = {
        "Success (No Error)": "#10b981",
        "Argument / Validation Error": "#ef4444",
        "Tool Selection Error": "#f59e0b",
        "Policy Constraint Violation": "#8b5cf6",
        "Premature Termination": "#ec4899"
    }

    for _, row in lat_profiles.iterrows():
        mode = row["error_mode"]
        d_vals = [row["drift_D1"], row["drift_D2"], row["drift_D3"]]
        col = colors.get(mode, "#64748b")
        ax1.plot([1, 2, 3], d_vals, marker="o", lw=2.2, label=f"{mode} (n={row['count']})", color=col)

    ax1.set_title("Latent ReAct: Drift Profiles by Failure Mode", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recursion Round ($r$)", fontsize=11)
    ax1.set_ylabel(r"Representation Drift ($D_r$)", fontsize=11)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["Round 1 ($D_1$)", "Round 2 ($D_2$)", "Round 3 ($D_3$)"])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")

    # Subplot B: Correlation Heatmap
    ax2 = axes[1]
    heat_df = df_corr[df_corr["architecture"] == "latent_react"].pivot(
        index="geometric_feature", columns="error_target", values="pearson_r"
    )
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2, cbar_kws={'label': 'Pearson $r$'})
    ax2.set_title("Geometric Metrics <-> Error Mode Correlations", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Geometric Metric", fontsize=11)
    ax2.set_xlabel("Error Type", fontsize=11)
    ax2.tick_params(axis="x", rotation=30)

    # Subplot C: ROC Curve for Error Inference Probing
    ax3 = axes[2]
    for arch, col in [("latent_react", "#2563eb"), ("text_react", "#f59e0b")]:
        d = roc_data[arch]
        ax3.plot(d["fpr"], d["tpr"], lw=2.5, color=col, label=f"{arch.replace('_', ' ').title()} (AUC = {d['auc']:.3f})")

    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    ax3.set_title("ROC Curve: Inferring Error from Latent Vector $H_r$", fontsize=12, fontweight="bold")
    ax3.set_xlabel("False Positive Rate", fontsize=11)
    ax3.set_ylabel("True Positive Rate", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc="lower right")

    plt.suptitle("Micro-Level Error Analysis & Latent Geometric Inference", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  [OK] Generated Error Analysis Figure: {PLOT_PATH}")

    # Copy to artifact directory
    art_dest = os.path.join(r"C:\Users\Admin\.gemini\antigravity-ide\brain\bdedc89f-02ac-4078-9cb2-4f17ab04cde9", "error_modes_vs_geometry.png")
    try:
        import shutil
        shutil.copy(PLOT_PATH, art_dest)
        print(f"  [OK] Copied to artifact directory: {art_dest}")
    except Exception as e:
        pass

    print("\nError Analysis & Predictive Probing complete!")


if __name__ == "__main__":
    run_error_geometry_analysis()
