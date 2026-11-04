"""
Stage 1 - Module 1: Latent Representation Extraction & Serialization Engine
Extracts high-dimensional hidden state representations (H_{a,r} in R^2560 and H_r in R^2560)
across all 912 benchmark task execution runs (r=0, 1, 2, 3 across Text ReAct & Latent ReAct).

Protocol:
  - Checkpoint 1 (Planner H1): Pooled state from Agent 1 (Planner)
  - Checkpoint 2 (Tool-Caller H2): Pooled state from Agent 2 (Tool-Caller & Observation)
  - Checkpoint 3 (Verifier H3): Pooled state from Agent 3 (Verifier / Feedback)
  - Consolidated Round State (H_r): Mean pooled representation over agent roles in R^2560
  - Normalization: L2 unit normalization for cosine similarity, drift, and SVD/PCA
"""
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

# Ensure common modules can be imported
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from common import (
    RESULTS_DIR,
    get_neural_adapters,
    text_to_latent_tensor,
)

LATENTS_DIR = os.path.join(RESULTS_DIR, "latents")
os.makedirs(LATENTS_DIR, exist_ok=True)
MASTER_CACHE_FILE = os.path.join(LATENTS_DIR, "all_representations.pt")


def extract_representation_for_round(
    round_data: Dict[str, Any],
    r_idx: int,
    adapters: Dict[str, Any],
    prev_feedback: torch.Tensor = None,
    arch: str = "latent_react",
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract H1, H2, H3 and consolidated Hr for a given round.
    Returns (H1, H2, H3, Hr, feedback_tensor).
    """
    with torch.no_grad():
        if arch == "latent_react":
            # 1. Agent 1: Latent Planner
            a1_data = round_data.get("agent_1_planner", {})
            prompt_p = a1_data.get("input_prompt", f"round_{r_idx}")
            h1_raw = text_to_latent_tensor(prompt_p, hidden_dim=2560, device=device)
            if prev_feedback is not None:
                h1_raw = h1_raw + prev_feedback
            h1_ref = adapters["reflector_inner"](h1_raw)
            h1_out = adapters["outer_rt"](h1_ref)  # shape (1, 32, 2560)
            H1 = h1_out.mean(dim=1).squeeze(0)    # shape (2560,)

            # 2. Agent 2: Tool-Caller & Environment Execution
            env_data = round_data.get("environment_execution", {})
            obs_text = env_data.get("observation_text", "")
            tc_data = round_data.get("agent_2_toolcaller", {})
            tc_prompt = tc_data.get("input_prompt", "")
            h2_raw = text_to_latent_tensor(f"{tc_prompt[:200]} {obs_text}", hidden_dim=2560, device=device)
            h2_ref = adapters["toolcaller_inner"](h2_raw)
            h2_out = adapters["outer_tr"](h2_ref)  # shape (1, 32, 2560)
            H2 = h2_out.mean(dim=1).squeeze(0)    # shape (2560,)

            # 3. Agent 3: Verifier
            a3_data = round_data.get("agent_3_verifier", {})
            v_text = a3_data.get("output_text", "")
            h3_raw = text_to_latent_tensor(v_text, hidden_dim=2560, device=device)
            h3_ref = adapters["reflector_inner"](h3_raw)
            H3 = h3_ref.mean(dim=1).squeeze(0)    # shape (2560,)
            feedback = h3_ref

        else:
            # Text ReAct Architecture
            # Map textual steps to shared representation space in R^2560
            a1_data = round_data.get("agent_1_planner", {})
            p_text = a1_data.get("thought_text", a1_data.get("input_prompt", f"plan_{r_idx}"))
            h1_raw = text_to_latent_tensor(p_text, hidden_dim=2560, device=device)
            H1 = adapters["reflector_inner"](h1_raw).mean(dim=1).squeeze(0)

            env_data = round_data.get("environment_execution", {})
            obs_text = env_data.get("observation_text", "")
            tc_data = round_data.get("agent_2_toolcaller", {})
            call_text = tc_data.get("output_text", "")
            h2_raw = text_to_latent_tensor(f"{call_text} {obs_text}", hidden_dim=2560, device=device)
            H2 = adapters["toolcaller_inner"](h2_raw).mean(dim=1).squeeze(0)

            a3_data = round_data.get("agent_3_verifier", {})
            critique_text = a3_data.get("output_text", "")
            h3_raw = text_to_latent_tensor(critique_text, hidden_dim=2560, device=device)
            H3 = adapters["reflector_inner"](h3_raw).mean(dim=1).squeeze(0)
            feedback = None

        # Consolidated Round Representation
        Hr = (H1 + H2 + H3) / 3.0

    return H1.cpu(), H2.cpu(), H3.cpu(), Hr.cpu(), feedback


def extract_all_trajectories():
    """Extract and cache representations for all 912 benchmark task runs."""
    print("=" * 80)
    print("STAGE 1: EXTRACTING HIGH-DIMENSIONAL LATENT REPRESENTATIONS (d_h = 2560)")
    print("=" * 80)
    t0 = time.time()

    adapters = get_neural_adapters()
    device = adapters["device"]

    jsonl_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "results_*.jsonl")))
    print(f"Found {len(jsonl_files)} result checkpoint files in {RESULTS_DIR}")

    # Deduplicate runs by (architecture, task_id, recursive_depth)
    all_runs = {}
    for fpath in jsonl_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    arch = record.get("architecture", "text_react")
                    tid = str(record.get("task_id", ""))
                    depth = int(record.get("recursive_depth", 0))
                    key = (arch, tid, depth)
                    all_runs[key] = record
                except Exception as e:
                    pass

    print(f"Total unique execution runs identified: {len(all_runs)}")

    extracted_database = {}
    processed_count = 0

    for key, record in all_runs.items():
        arch, tid, depth = key
        tier = record.get("complexity_tier", "easy")
        is_success = bool(record.get("is_success", False))
        accuracy = float(record.get("Accuracy", 0.0))
        trajectories = record.get("round_trajectories", [])

        round_representations = {}
        prev_feedback = None

        for r_idx, r_data in enumerate(trajectories):
            H1, H2, H3, Hr, feedback = extract_representation_for_round(
                r_data, r_idx, adapters, prev_feedback=prev_feedback, arch=arch, device=device
            )
            prev_feedback = feedback
            round_representations[r_idx] = {
                "H1": H1,
                "H2": H2,
                "H3": H3,
                "Hr": Hr
            }

        task_tensor_entry = {
            "task_id": tid,
            "architecture": arch,
            "complexity_tier": tier,
            "recursive_depth": depth,
            "is_success": is_success,
            "Accuracy": accuracy,
            "num_rounds": len(trajectories),
            "rounds": round_representations
        }

        extracted_database[key] = task_tensor_entry

        # Save individual task tensor file
        task_out_file = os.path.join(LATENTS_DIR, f"{arch}_task_{tid}_r{depth}.pt")
        torch.save(task_tensor_entry, task_out_file)

        processed_count += 1
        if processed_count % 100 == 0 or processed_count == len(all_runs):
            print(f"  Processed [{processed_count}/{len(all_runs)}] runs | Elapsed: {time.time() - t0:.1f}s")

    # Save consolidated master cache
    print(f"\nSerializing master representations file to: {MASTER_CACHE_FILE} ...")
    torch.save(extracted_database, MASTER_CACHE_FILE)
    elapsed = time.time() - t0

    file_size_mb = os.path.getsize(MASTER_CACHE_FILE) / (1024 * 1024)
    print(f"Extraction complete! {len(extracted_database)} task runs cached ({file_size_mb:.2f} MB) in {elapsed:.2f}s.")
    return extracted_database


if __name__ == "__main__":
    extract_all_trajectories()
