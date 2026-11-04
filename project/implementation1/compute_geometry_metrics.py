"""
Stage 1 - Module 2: Geometric Computation Engine & Cohort Stratification
Calculates all 7 geometric metrics across all 912 benchmark task execution runs:
  1. Directional Alignment (Cosine Similarity)
  2. Representation Drift (D_r = 1 - CosSim(H_r, H_{r-1})) and Drift Decay Slope
  3. Inter-Agent Representation Distance (D_ij = ||H_i - H_j||_2 for Planner, Tool-Caller, Verifier)
  4. Effective Rank (ER) via SVD of centered representation matrices
  5. Principal Subspace Angles (theta_k) between Success and Failure manifolds
  6. Success-Failure Centroid Separation (Delta_SF = ||H_succ - H_fail||_2)
  7. Low-Dimensional PCA Projections (PC1, PC2, PC3) for phase-space trajectory tracking

Categorizes tasks into 4 behavioral cohorts:
  - Cohort 1: Instant Success (r=0 pass)
  - Cohort 2: Recovered Success (r=0 fail -> r>0 pass) [Primary focus of Hypothesis H5]
  - Cohort 3: Persistent Failure (fail across all r)
  - Cohort 4: Degraded / Oscillation (r=0 pass -> later fail)

Exports:
  - results/geometric_analysis_metrics.csv
  - results/trajectory_cohort_summary.csv
  - results/subspace_angles_and_separability.json
"""
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.linalg as la
from scipy import stats
import torch
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LATENTS_DIR = os.path.join(RESULTS_DIR, "latents")
MASTER_CACHE_FILE = os.path.join(LATENTS_DIR, "all_representations.pt")
METRICS_CSV_FILE = os.path.join(RESULTS_DIR, "geometric_analysis_metrics.csv")
COHORT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "trajectory_cohort_summary.csv")
SUBSPACE_JSON_FILE = os.path.join(RESULTS_DIR, "subspace_angles_and_separability.json")


def compute_effective_rank(matrix: np.ndarray) -> float:
    """
    Compute Effective Rank ER(H) = exp(-sum p_k * ln(p_k))
    where p_k = sigma_k / sum(sigma) from SVD of centered matrix.
    """
    if matrix.shape[0] < 2:
        return 1.0
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        s = s[s > 1e-12]
        if len(s) == 0:
            return 1.0
        p = s / np.sum(s)
        entropy = -np.sum(p * np.log(p + 1e-12))
        return float(np.exp(entropy))
    except Exception:
        return 1.0


def compute_principal_angles(subspace_u: np.ndarray, subspace_v: np.ndarray) -> List[float]:
    """
    Compute principal angles theta_k in degrees between orthonormal bases U and V.
    cos(theta_k) = singular values of (U.T @ V).
    """
    try:
        q_u, _ = np.linalg.qr(subspace_u)
        q_v, _ = np.linalg.qr(subspace_v)
        m = q_u.T @ q_v
        _, s, _ = np.linalg.svd(m)
        s = np.clip(s, -1.0, 1.0)
        angles_rad = np.arccos(s)
        angles_deg = np.rad2deg(angles_rad)
        return [float(a) for a in angles_deg]
    except Exception:
        return [0.0]


def run_geometric_analysis():
    print("=" * 80)
    print("STAGE 1: COMPUTING GEOMETRIC METRICS & COHORT STRATIFICATION")
    print("=" * 80)
    t0 = time.time()

    if not os.path.exists(MASTER_CACHE_FILE):
        raise FileNotFoundError(f"Master cache not found at {MASTER_CACHE_FILE}. Run extraction first.")

    print(f"Loading master representations from: {MASTER_CACHE_FILE} ...")
    database = torch.load(MASTER_CACHE_FILE, map_location="cpu", weights_only=False)
    print(f"Successfully loaded {len(database)} task execution representations.")

    # Group runs by (architecture, task_id) across depths r=0, 1, 2, 3
    tasks_by_arch = {"latent_react": {}, "text_react": {}}
    for key, data in database.items():
        arch, tid, depth = key
        if arch not in tasks_by_arch:
            tasks_by_arch[arch] = {}
        if tid not in tasks_by_arch[arch]:
            tasks_by_arch[arch][tid] = {}
        tasks_by_arch[arch][tid][depth] = data

    # Collect all H_r representations for global PCA fit
    all_hr_vectors = []
    all_metadata = []
    for arch in ["latent_react", "text_react"]:
        for tid, depth_dict in tasks_by_arch[arch].items():
            for depth, record in depth_dict.items():
                rounds = record.get("rounds", {})
                for r_idx, r_tensors in rounds.items():
                    hr = r_tensors["Hr"].numpy()
                    all_hr_vectors.append(hr)
                    all_metadata.append((arch, tid, depth, r_idx))

    all_hr_mat = np.array(all_hr_vectors)
    print(f"Fitting Global PCA on representation matrix {all_hr_mat.shape} ...")
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(all_hr_mat)
    explained_var = pca.explained_variance_ratio_
    print(f"PCA Fitted: Top 3 PC Variance = [{explained_var[0]*100:.2f}%, {explained_var[1]*100:.2f}%, {explained_var[2]*100:.2f}%]")

    # Map PCA coordinates back
    pca_lookup = {}
    for idx, meta in enumerate(all_metadata):
        pca_lookup[meta] = pca_coords[idx]

    # Process task trajectories across depth
    task_rows = []
    cohort_stats = {
        "latent_react": {"Instant Success": 0, "Recovered Success": 0, "Persistent Failure": 0, "Degraded": 0},
        "text_react": {"Instant Success": 0, "Recovered Success": 0, "Persistent Failure": 0, "Degraded": 0}
    }

    # For Subspace calculations
    success_subspace_vectors = {"latent_react": [], "text_react": []}
    failure_subspace_vectors = {"latent_react": [], "text_react": []}

    for arch in ["latent_react", "text_react"]:
        for tid, depth_dict in sorted(tasks_by_arch[arch].items()):
            # Determine trajectory cohort across available depths (0..3)
            r0_rec = depth_dict.get(0)
            r3_rec = depth_dict.get(3, depth_dict.get(max(depth_dict.keys())))
            tier = r3_rec.get("complexity_tier", "easy") if r3_rec else "easy"

            succ_0 = r0_rec.get("is_success", False) if r0_rec else False
            succ_1 = depth_dict.get(1, {}).get("is_success", False)
            succ_2 = depth_dict.get(2, {}).get("is_success", False)
            succ_3 = r3_rec.get("is_success", False) if r3_rec else False

            any_later_succ = succ_1 or succ_2 or succ_3

            if succ_0 and succ_3:
                cohort = "Instant Success"
            elif (not succ_0) and any_later_succ:
                cohort = "Recovered Success"
            elif (not succ_0) and (not any_later_succ):
                cohort = "Persistent Failure"
            else:
                cohort = "Degraded"

            cohort_stats[arch][cohort] += 1

            # Focus on depth=3 trajectory (which executed rounds 0, 1, 2, 3)
            traj_record = depth_dict.get(3, r3_rec)
            rounds = traj_record.get("rounds", {})
            r_keys = sorted(rounds.keys())

            # Extract vectors for each round
            hr_by_r = {}
            h1_by_r, h2_by_r, h3_by_r = {}, {}, {}
            for r_idx in r_keys:
                hr_by_r[r_idx] = rounds[r_idx]["Hr"].numpy()
                h1_by_r[r_idx] = rounds[r_idx]["H1"].numpy()
                h2_by_r[r_idx] = rounds[r_idx]["H2"].numpy()
                h3_by_r[r_idx] = rounds[r_idx]["H3"].numpy()

            # 1. Cosine Similarities & Representation Drift Dr = 1 - CosSim(Hr, Hr-1)
            dr_values = {}
            cos_values = {}
            for r in range(1, len(r_keys)):
                h_prev = hr_by_r[r - 1]
                h_curr = hr_by_r[r]
                norm_p = np.linalg.norm(h_prev) + 1e-12
                norm_c = np.linalg.norm(h_curr) + 1e-12
                c_sim = float(np.dot(h_prev, h_curr) / (norm_p * norm_c))
                drift = float(1.0 - c_sim)
                cos_values[f"cos_sim_r{r}_r{r-1}"] = round(c_sim, 5)
                dr_values[f"drift_D{r}"] = round(drift, 5)

            d1 = dr_values.get("drift_D1", 0.0)
            d2 = dr_values.get("drift_D2", 0.0)
            d3 = dr_values.get("drift_D3", 0.0)
            drift_slope = round((d3 - d1) / 2.0, 5)  # Negative indicates monotonic decay

            # 2. Inter-Agent Distances D_ij = ||Hi - Hj||_2
            inter_agent_dists = []
            for r in r_keys:
                d12 = float(np.linalg.norm(h1_by_r[r] - h2_by_r[r]))
                d23 = float(np.linalg.norm(h2_by_r[r] - h3_by_r[r]))
                d31 = float(np.linalg.norm(h3_by_r[r] - h1_by_r[r]))
                inter_agent_dists.append((d12 + d23 + d31) / 3.0)
            mean_inter_agent_dist = round(float(np.mean(inter_agent_dists)), 4) if inter_agent_dists else 0.0

            # 3. Low-Dimensional PCA coordinates
            pca_r0 = pca_lookup.get((arch, tid, 3, 0), [0.0, 0.0, 0.0])
            pca_r3 = pca_lookup.get((arch, tid, 3, max(r_keys)), [0.0, 0.0, 0.0])

            # Subspace vector collection (final round state H_final)
            final_h = hr_by_r[max(r_keys)]
            if succ_3:
                success_subspace_vectors[arch].append(final_h)
            else:
                failure_subspace_vectors[arch].append(final_h)

            task_rows.append({
                "task_id": tid,
                "architecture": arch,
                "complexity_tier": tier,
                "cohort": cohort,
                "is_success_r0": succ_0,
                "is_success_r3": succ_3,
                "is_recovered": (cohort == "Recovered Success"),
                "drift_D1": d1,
                "drift_D2": d2,
                "drift_D3": d3,
                "drift_decay_slope": drift_slope,
                "cos_sim_r1_r0": cos_values.get("cos_sim_r1_r0", 0.0),
                "cos_sim_r2_r1": cos_values.get("cos_sim_r2_r1", 0.0),
                "cos_sim_r3_r2": cos_values.get("cos_sim_r3_r2", 0.0),
                "inter_agent_distance": mean_inter_agent_dist,
                "pca_r0_pc1": round(float(pca_r0[0]), 4),
                "pca_r0_pc2": round(float(pca_r0[1]), 4),
                "pca_r3_pc1": round(float(pca_r3[0]), 4),
                "pca_r3_pc2": round(float(pca_r3[1]), 4),
            })

    # 4. Effective Rank ER across tiers, cohorts, and rounds
    print("\nComputing Effective Rank (ER) singular value spectrums ...")
    er_metrics = {}
    for arch in ["latent_react", "text_react"]:
        for r_idx in range(4):
            r_vecs = [
                database[(arch, str(row["task_id"]), 3)]["rounds"][r_idx]["Hr"].numpy()
                for row in task_rows if row["architecture"] == arch and (arch, str(row["task_id"]), 3) in database
            ]
            if len(r_vecs) > 2:
                er_val = compute_effective_rank(np.array(r_vecs))
                er_metrics[f"{arch}_ER_round_{r_idx}"] = round(er_val, 3)
                print(f"  {arch} Round {r_idx} Effective Rank: {er_val:.2f}")

    # Add round-specific ER to each row
    for row in task_rows:
        arch = row["architecture"]
        row["ER_r0"] = er_metrics.get(f"{arch}_ER_round_0", 1.0)
        row["ER_r1"] = er_metrics.get(f"{arch}_ER_round_1", 1.0)
        row["ER_r2"] = er_metrics.get(f"{arch}_ER_round_2", 1.0)
        row["ER_r3"] = er_metrics.get(f"{arch}_ER_round_3", 1.0)

    # 5. Principal Subspace Angles & Centroid Separability
    subspace_results = {}
    for arch in ["latent_react", "text_react"]:
        succ_mat = np.array(success_subspace_vectors[arch])
        fail_mat = np.array(failure_subspace_vectors[arch])

        if len(succ_mat) > 3 and len(fail_mat) > 3:
            # Centroid separation
            mean_succ = np.mean(succ_mat, axis=0)
            mean_fail = np.mean(fail_mat, axis=0)
            delta_sf = float(np.linalg.norm(mean_succ - mean_fail))

            # Principal angles between top 5 principal components
            pca_s = PCA(n_components=min(5, len(succ_mat))).fit(succ_mat)
            pca_f = PCA(n_components=min(5, len(fail_mat))).fit(fail_mat)
            angles = compute_principal_angles(pca_s.components_.T, pca_f.components_.T)

            subspace_results[arch] = {
                "success_failure_centroid_separation": round(delta_sf, 4),
                "principal_angles_deg": [round(a, 2) for a in angles],
                "theta_1_deg": round(angles[0], 2) if angles else 0.0
            }
            print(f"  {arch} Success vs Failure Separation: {delta_sf:.4f} | Principal Angle theta_1: {angles[0]:.2f} deg")

    # Save Metrics CSV
    fieldnames = list(task_rows[0].keys())
    with open(METRICS_CSV_FILE, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(task_rows)
    print(f"\nSaved consolidated metrics to: {METRICS_CSV_FILE}")

    # Save Cohort Summary CSV
    cohort_summary_rows = []
    for arch in ["latent_react", "text_react"]:
        for cohort, cnt in cohort_stats[arch].items():
            matching_rows = [r for r in task_rows if r["architecture"] == arch and r["cohort"] == cohort]
            mean_d1 = np.mean([r["drift_D1"] for r in matching_rows]) if matching_rows else 0.0
            mean_d3 = np.mean([r["drift_D3"] for r in matching_rows]) if matching_rows else 0.0
            mean_slope = np.mean([r["drift_decay_slope"] for r in matching_rows]) if matching_rows else 0.0
            cohort_summary_rows.append({
                "architecture": arch,
                "cohort": cohort,
                "task_count": cnt,
                "percentage": round(cnt / 114.0 * 100.0, 1),
                "mean_drift_D1": round(float(mean_d1), 5),
                "mean_drift_D3": round(float(mean_d3), 5),
                "mean_decay_slope": round(float(mean_slope), 5)
            })

    with open(COHORT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f_cs:
        writer = csv.DictWriter(f_cs, fieldnames=list(cohort_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cohort_summary_rows)
    print(f"Saved cohort summary to: {COHORT_SUMMARY_CSV}")

    # Save Subspace JSON
    with open(SUBSPACE_JSON_FILE, "w", encoding="utf-8") as f_sub:
        json.dump(subspace_results, f_sub, indent=2)
    print(f"Saved subspace angles to: {SUBSPACE_JSON_FILE}")

    # Hypothesis H5 Statistical Testing
    print("\n" + "=" * 80)
    print("HYPOTHESIS H5 STATISTICAL VALIDATION (WELCH'S T-TEST)")
    print("=" * 80)
    for arch in ["latent_react", "text_react"]:
        recov_slopes = [r["drift_decay_slope"] for r in task_rows if r["architecture"] == arch and r["cohort"] == "Recovered Success"]
        fail_slopes = [r["drift_decay_slope"] for r in task_rows if r["architecture"] == arch and r["cohort"] == "Persistent Failure"]

        if recov_slopes and fail_slopes:
            t_stat, p_val = stats.ttest_ind(recov_slopes, fail_slopes, equal_var=False)
            print(f"  [{arch.upper()}] Drift Decay Slope: Recovered Success ({np.mean(recov_slopes):.5f}) vs Persistent Failure ({np.mean(fail_slopes):.5f})")
            print(f"    Welch's t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
            if p_val < 0.05:
                print(f"    => STATISTICALLY SIGNIFICANT (p < 0.05)! Hypothesis H5 CONFIRMED for {arch}!")
            else:
                print(f"    => p >= 0.05")

    print(f"\nAll Stage 1 geometric calculations completed in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    run_geometric_analysis()
